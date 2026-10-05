import json, os
import re
from typing import Optional, List, Union
from dotenv import load_dotenv
import pandas as pd
import webbrowser
import html as html_lib
from datetime import datetime, date

from google.adk import Agent, Workflow, Event, Context
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


from .classes import EmailExtraction, OfferValidation, GroupSelection
from .product_lookup import search_by_description_openai
from .create_offer import build_grouped_offer
from .node_offer import create_proposals
from .render_html import _write_and_open_preview, PREVIEW_IN_BROWSER, llm_text_to_html
from .finding_prices import price_order



load_dotenv()

 
SYSTEM_EXTRACTION = """Jesteś asystentem, który wyciąga ustrukturyzowane dane z polskich e-maili
biznesowych (B2B) zapytań o kosze prezentowe firmy Fabulosa. Zwracasz TYLKO JSON
zgodny z podanym schematem (products, deadline, policy_questions, intent).
 
Zasady:
- products: lista produktów, o które pyta klient. KAŻDY produkt ma własne ograniczenia
  (cena, alkohol, dostawa), bo w jednym mailu różne produkty mogą mieć różne warunki
  (np. "30 szt. do 150 zł i 10 szt. do 200 zł" -> dwa produkty z różnym price_max).
  Jeśli klient poda JEDNO ograniczenie dla całości -> zastosuj je do każdego produktu.
  Dla każdego produktu:
  - url: jeśli jest link do fabulosa.pl -> tutaj. Inaczej null.
  - name: jeśli klient podaje dokładną nazwę/kod (np. "BJ89", "PL44 Czas na piernik").
  - description: jeśli klient tylko OPISUJE czego chce (np. "coś z kawą i słodyczami").
    NIGDY nie umieszczaj URL-a w description.
  - quantity: liczba sztuk (int) lub null.
  - alcohol: "none" bez alkoholu; "required" z alkoholem (dowolnym);
    "specific" konkretny alkohol (wtedy alcohol_detail, np. "jack daniels 0,7l"); "any" gdy nie wspomina.
  - price_basis: "netto" lub "brutto". Jeśli klient NIE precyzuje -> "brutto".
  - price_per: "person" gdy "na osobę", "total" gdy łączna kwota, inaczej "piece".
  - price_includes_delivery: true jeśli budżet jest "z wysyłką".
  - delivery: "to_addresses" (na wiele adresów), "pickup" (odbiór własny),
    "to_client_warehouse" (do magazynu klienta), "unspecified".
- deadline (na poziomie całego maila): data YYYY-MM-DD jeśli podana, inaczej null.
- price_min, price_max -> w przypadku kiedy klient poda zakres 160-170 zł to zapisz price_min = 160 i price_max = 170.
- policy_questions: tematy, o które klient PYTA (nie odpowiadaj na nie):
  "discount", "delivery_time", "different_addreses", "shipping_cost", "personalization",
  "greeting_card", "international", "payment", "other".
- Jeżeli klient pyta o dołączenie kartki z życzeniami to zaznacz "greeting_card", nie "personalization".
- Jeżeli klient pyta o dostawy na wiele adresów to zaznacz to zarówno w policy_questions jak i delivery.
- intent — decide in this order:
  1. "order" when the client lists SPECIFIC catalog products (code like PL44/MCX54,
     exact name, or link) WITH quantities. This is "order" EVEN IF the client:
     - writes "jestem zainteresowana", "chciałabym", "proszę o wycenę",
     - asks for the price, shipping cost, lead time, several addresses or a discount.
     Questions about price/logistics go to policy_questions; they do not change intent.
  2. "inquiry" when the client DESCRIBES what they want or asks for a proposal
     ("proszę o propozycję", "poszukuję...", "coś podobnego do PL44", "w stylu SU42"),
     EVEN IF they use the word "zamówić". Products given only as a reference
     ("podobne do", "w stylu", "coś jak") mean "inquiry", not "order".
  3. "browse" when the client only asks about the offer or possibilities in general,
     with no specific requirements.
- Jeśli czegoś nie ma w mailu, użyj null / "any" / pustej listy.
- Jeśli klient prosi o coś "podobnego do" i wymienia konkretne produkty (linki/nazwy),
  potraktuj KAŻDY z tych produktów jako osobny produkt referencyjny (url lub name),
  z tymi samymi ograniczeniami (cena, alkohol, kolor). NIE zwijaj ich w jeden ogólny opis.
- Jeśli klient podaje różne ilości dla różnych ADRESÓW lub SPOSOBÓW dostawy
  (np. "19 zestawów na adres i 11 do magazynu"), to jest podział LOGISTYCZNY, nie produktowy.
  NIE twórz osobnych produktów — użyj SUMY jako quantity, a szczegóły dostawy zostaw człowiekowi.
  Dziel na osobne produkty TYLKO gdy różnią się CENĄ lub RODZAJEM (np. "12 do 150 zł i 2 do 200 zł").
- price_includes_delivery: true jeśli budżet jest podany "z wysyłką" / "już z wysyłką" / "z dostawą".
  Jeśli klient podaje JEDEN budżet z wysyłką dla całości, ustaw True dla WSZYSTKICH produktów,
  niezależnie od sposobu dostawy.
- Jezeli klient pyta się o dostawę i wspomni o zagranicznych to zawsze dopisz do policy_questions 'international_delivery' "
"""

INSTRUCTION_EXTRACTION = SYSTEM_EXTRACTION 


URL_RE = re.compile(r"https?://fabulosa\.pl/\S*?/(\d+)", re.IGNORECASE)
 
 
def postprocess_extraction(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    if not llm_response.content or not llm_response.content.parts:
        return None
 
    raw_text = llm_response.content.parts[0].text
    if not raw_text:
        return None
 
    try:
        data = EmailExtraction.model_validate_json(raw_text)
    except Exception 
        return None
 
    email_text = callback_context.state.get("email", "")
 
    for p in data.products:
        if p.url:
            m = URL_RE.search(p.url)
            p.product_id = int(m.group(1)) if m else None
        else:
            p.product_id = None
        if p.description:
            cleaned = re.sub(r"https?://\S+", "", p.description).strip()
            p.description = cleaned or None
 
    llm_response.content.parts[0].text = data.model_dump_json()
    return llm_response


def normalise(node_input : str):
    return Event(state = {"clients_email" : node_input, "todays_date" : datetime.now()}, output= node_input)


extraction_agent = Agent(name = 'email_extractor',
                         description= 'Extracts structured information about ordered products etc. from the customers emails.',
                         model= "gemini-3.5-flash",
                         instruction= INSTRUCTION_EXTRACTION,
                         after_model_callback= postprocess_extraction,
                         output_schema= EmailExtraction)

policy_info = {
    "discount": "W trakcie sezonu świątecznego rabaty udzielane są od kwoty 10 tysięcy zł netto.",
    "delivery_time": "Czas realizacji zamówienia wynosi do 8 dni roboczych po zaksięgowaniu wpłaty.",
    "personalization": "Istnieje możliwość personalizacji zamówionych produktów. Logujemy pudełka, produkty, wstążki, drukujemy bileciki z życzeniami czy logo. Koszt jest zależny od ilości, metody znakowania i rodzaju logowania.",
    "greeting_card": "Istnieje możliwość dołączenia kartki świątecznej z życzeniami czy logo. Możemy zrobić dla Państwa projekt według wytycznych lub możemy dołączyć do zestawów kartki od Państwa.",
    "international_delivery": "Cena wysyłki paczek zagranicznych jest kalkulowana osobno po otrzymaniu dokładnego adresu. Zastrzegamy sobie odmowy wysyłki paczek poza UE.",
    "payment": "Istnieje możliwość rozliczenia transakcji za pomocą przelewu na konto bankowe, za pobraniem przy odbiorze zamówienia oraz za pomocą przelewu online Przelewy24.",
    "different_addreses": "Jest możliwość wysyłania paczek na wskazane przez Państwa adresy. Wysyłamy wtedy do Państwa plik Excela w celu uzupełnienia adresów.",
}

MIN_QTY_HIGH_SEASON = 10
HIGH_SEASON_MONTHS = {9, 10, 11, 12, 1, 2}

def shipping_note(total_qty: int) -> str:
    if total_qty > 12:
        return f"Zamówienie obejmuje łącznie {total_qty} szt., dlatego koszt dostawy ustalamy indywidualnie."
    if total_qty >= 7:
        return f"Dla {total_qty} szt.: 40 zł przy płatności przelewem lub 45 zł za pobraniem."
    return f"Dla {total_qty} szt.: 20 zł przy płatności przelewem lub 26,50 zł za pobraniem."



def intent_router(node_input : EmailExtraction, ctx: Context):
    policy_questions_asked = node_input.policy_questions
    res = {}

    for question in policy_questions_asked:
        res[question] = policy_info[question]

    products = node_input.products   
    
    total = sum(int(p.quantity) for p in products if p.quantity)

    too_small = None
    if total:
        too_small = [p for p in products if int(p.quantity) < MIN_QTY_HIGH_SEASON]
    res["shipping_note"] = shipping_note(total)

    if date.today().month in HIGH_SEASON_MONTHS and too_small:
        items = ", ".join(f'{p.description} ({p.quantity} szt.)' for p in too_small)
        res["min_qty_note"] = f"W sezonie świątecznym przyjmujemy zamówienia na jeden produkt od {MIN_QTY_HIGH_SEASON} szt. Dotyczy: {items}."
        if node_input.intent == "order":
            payload = [p.model_dump(mode="json", exclude_none=True) for p in products]
            return Event(route = node_input.intent, output= json.dumps(payload, ensure_ascii=False), state = {"policy_info" : res})
        else:
            return Event(route = node_input.intent, output= node_input, state = {"policy_info" : res})
    
    else:
        res["min_qty_note"] = None
        if node_input.intent == "order":
            payload = [p.model_dump(mode="json", exclude_none=True) for p in products]
            return Event(route = node_input.intent, output= json.dumps(payload, ensure_ascii=False), state = {"policy_info" : res})
        else:
            return Event(route = node_input.intent, output= node_input, state = {"policy_info" : res})





validator_agent = Agent(name = 'validator_agent',
                        description= 'Takes product proposals and validates and returns the best ones',
                        model= "gemini-3.1-flash-lite",
                        instruction= """Jesteś asystentem weryfikującym trafność propozycji produktów w ofercie dla klienta.

Otrzymujesz listę grup ("groups"), każda z osobnym "group_index" (liczbą całkowitą, zaczynając od 0).
Każda grupa zawiera:
- "group_index": numer grupy — UŻYWAJ WYŁĄCZNIE TEGO do identyfikacji grupy w odpowiedzi.
  Różne grupy mogą mieć IDENTYCZNY "label" (to samo zapytanie klienta podzielone na
  różne ilości) — nigdy nie polegaj na "label" jako identyfikatorze.
- "label": czego dokładnie szukał klient dla tej grupy.
- "quantity": ile sztuk klient potrzebuje.
- "product_description": słownik { kod_produktu: {"name", "description", "price_net"} } —
  kandydaci znalezieni przez wyszukiwarkę dla tej grupy.

Twoje zadanie: dla KAŻDEJ grupy sprawdź KAŻDEGO kandydata i zdecyduj, czy jego RZECZYWISTA
zawartość (pole "description") faktycznie odpowiada temu, o co prosił klient w "label" —
nie tylko czy nazwa produktu brzmi podobnie.

Zasady:
1. Analizuj wyłącznie "description" i "name" — nie zgaduj i nie dodawaj informacji,
   których tam nie ma.
2. Zwracaj WYŁĄCZNIE kody, które faktycznie widziałeś w "product_description" DLA TEJ
   KONKRETNEJ grupy. Nigdy nie wymyślaj kodów ani nie przenoś kodu z innej grupy.
3. Jeśli klient wymagał braku alkoholu ("bezalkoholowy"), a zawartość produktu wskazuje
   na alkohol (wino, whisky, nalewka, likier itp.) — odrzuć ten kod, nawet jeśli nazwa
   sugeruje coś przeciwnego.
4. Jeśli klient wymagał konkretnego składnika (konkretny alkohol, kawa, herbata,
   określony kolor itp.) — zaakceptuj tylko produkty, które faktycznie go zawierają
   według opisu.
5. "price_net" jest już wstępnie przefiltrowana wcześniej w procesie — traktuj ją jako
   dodatkowy kontekst, NIE jako główne kryterium. Nie odrzucaj produktu wyłącznie z
   powodu ceny, chyba że wygląda na oczywisty błąd dopasowania.
6. Zawsze musisz zwrócić minimalnie jeden produkt - jezeli coś lekko nie spełnia wymaganych oczekiwań to napisz w notatce czego nie spełnia. 
7. Często kleinci będą mówili o świątecznych koszach prezentowych. Jezeli klienci nie mówią bezpośrednio o wielkanocnych koszach to zawsze zakładaj bozonarodzeninowe. Jezeli nic nie jest powiedziane o wielkanocy to ich nie proponuj.

Dodaj swój tok myślowy dlaczego zaakceptowałeś albo odrzuciłeś kazdy produkt. Max jedno zdanie na jeden produkt.
""",
                        output_schema=OfferValidation)

def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def dedupe_selections(
    selections: List[Union[GroupSelection, dict]],
    drop_empty: bool = False,
    dedupe_codes_across_groups: bool = True,
) -> List[GroupSelection]:
    """
    Remove duplicate selections (same label + quantity + set of codes),
    keeping the first occurrence.

    - dedupe_codes_across_groups: a code already offered in an earlier group is
      removed from later groups; a group left with no codes this way is dropped.
    - drop_empty: also drop groups that had no codes to begin with.
    """
    result: List[GroupSelection] = []
    seen_keys: set[tuple] = set()
    seen_codes: set[str] = set()

    for raw in selections:
        sel = GroupSelection.model_validate(raw)
        codes = list(dict.fromkeys(sel.codes))  # dedupe codes, keep order
        key = (_norm(sel.label), sel.quantity, frozenset(codes))

        if key in seen_keys:
            continue
        seen_keys.add(key)

        had_codes = bool(codes)
        if dedupe_codes_across_groups:
            codes = [c for c in codes if c not in seen_codes]

        if not codes and (had_codes or drop_empty):
            continue

        seen_codes.update(codes)
        result.append(sel.model_copy(update={"codes": codes}))

    return result

def save_context(node_input : OfferValidation, ctx : Context):
    """ Here we have a function thatt saves into the state proposals as a context for the response agent. """
    proposals = ctx.state.get('proposals')
    i = 0
    selections_deduped = dedupe_selections(node_input.selections)
    print(selections_deduped)
    for selection in selections_deduped:
        list_codes = selection.codes
        pd = proposals[i].get("product_description") or {}
        codes_proposals = list(pd.keys()) if isinstance(pd, dict) else []
        for code in codes_proposals:
            if not code in list_codes:
                del proposals[i]["product_description"][code]

        i += 1
    return Event(state = {'proposals' : proposals, "for_offer" : node_input})



OFFER_TOKEN = "{Offer here}"

INSTRUCTION_RESPONSE_INQUIRY = f"""Jesteś pracownikiem działu handlowego. Piszesz odpowiedź e-mail do klienta B2B
po polsku, w liczbie mnogiej ("przygotowaliśmy", "zapraszamy"), rzeczowo i uprzejmie.

## Dane wejściowe
Mail klienta:
{{clients_email}}

Podsumowanie oferty (tylko do orientacji; nie przepisuj z niego żadnych wartości):
{{proposals}}

Fakty wyliczone przez system (są pewne; przekaż je klientowi, niczego nie licz sam):
- Minimalna ilość: {{policy_info['min_qty_note']}}
- Dostawa dla tego zamówienia JEZELI klient zamawia wszystko na jeden adres: {{policy_info['shipping_note']}}. Jezeli klient zamawia na wiele adresów to nie pisz tego nigdy 

Jezeli mamy null w któryś z powyzszych rubryk to nie wspominaj o nich. 

Polityka firmy (tylko do pytań, których nie obejmują fakty powyżej):
{{policy_info}}
Zawsze korzystaj ze wszystkich elementów zawartych powyzej! Jezeli zostały one dodane to oznacza to, ze klient o nie zapytał. 

## Struktura maila (dokładnie w tej kolejności)
1. "Dzień dobry,"
2. 1–2 zdania: podziękowanie za kontakt i nawiązanie do zapytania.
3. Osobna linia zawierająca wyłącznie: {OFFER_TOKEN}
4. Jeśli "Minimalna ilość" jest inna niż "brak" — przekaż tę informację w jednym-dwóch zdaniach.
5. Odpowiedzi na pytania klienta dotyczące zasad (dostawa, płatność, terminy itp.) —
   tylko na pytania, które klient faktycznie zadał. O dostawie pisz wyłącznie na podstawie
   "Dostawa dla tego zamówienia". Jeśli klient o nic nie pytał, pomiń ten punkt.
   W przypadku jezeli klient pyta się o termin realizacji to pamiętaj 
6. Jedno zdanie zachęty do kontaktu, następnie "Z poważaniem," i w nowej linii "Dział Handlowy".
7. Jezeli odpowiadasz na kilka pytań z policy questions to pisz je w oddzielnych akapitach, zeby wszystko było bardziej czytelne 
8. Jezeli piszesz o

## Zakazy
- Nie podawaj cen produktów, nazw, kodów ani linków — system wstawi je w miejsce tokenu.
- Nie pisz, że oferta zostanie przesłana później — oferta jest częścią tego maila.
- Nie wymyślaj informacji spoza danych wejściowych. Jeśli czegoś nie wiesz, napisz,
  że doprecyzujemy to w dalszym kontakcie.
- Zwróć wyłącznie treść maila: bez tematu, bez markdown, bez cudzysłowów.
"""

"""Piszesz odpowiedź e-mail do klienta B2B w języku polskim, w tonie
rzeczowym i uprzejmym, tak jak odpowiada dział handlowy firmy - pisz w liczbie mnogiej!
 
Masz dostęp do:
- treści maila klienta (stan: {{clients_email}})
- skróconego podsumowania oferty (stan: proposals) — lista grup, każda
  z label i quantity, BEZ cen i kodów produktów (te jeszcze nie istnieją na
  tym etapie). Używaj go WYŁĄCZNIE do tego, żeby wiedzieć z grubsza, o czym
  jest oferta (ile grup, czego dotyczą) — nie przepisuj z niego żadnych
  konkretnych wartości do treści maila, bo nie ma tam żadnych do przepisania.
 
Zasady:
1. Napisz krótkie, naturalne wprowadzenie (nawiązanie do zapytania klienta,
   podziękowanie za kontakt).
2. W miejscu, gdzie ma się pojawić właściwa oferta z produktami, cenami i
   zdjęciami, wstaw DOKŁADNIE ten token, bez żadnych zmian i bez własnego
   tekstu wokół niego w tej samej linii: {OFFER_TOKEN}
8. ZAWSZE zamówienia w sezonie wysokim (zima/jesień - jaka jest pora sprawdź w {{todays_date}} ) na jeden   produkt są przyjmowane od 10 sztuk. Jezeli klient poda jakiś produkt z ilością mniejsza od 10 w sezonie     wysokim to napisz, ze przyjmujemy zamówienia na jeden produkt wyłącznie od 10 sztuk. 

3. Jezeli chodzi o pytania dotyczące polityki firmy to skorzystaj z {{policy_info}} tam znajdziesz odpowiedzi na zadane pytania przez klientów. Dodatkowo mozesz wnioskować z tych informacji. Na przykład gdy klient pyta się o cenę dostawy to sprawdź w jakiej ilości zostało złozone zamówienie albo o jaką ilość pytał się klient i nie dawaj informacji o wszystkich mozliwościach tylko o tym co będzie dotyczyło klienta.
   Jeśli nie było żadnych pytań o politykę pomiń.
4. Napisz krótkie zakończenie (zachęta do kontaktu, pozdrowienie, zwrot
   grzecznościowy).
5. Nie generuj żadnych cen, nazw produktów, kodów, ani linków samodzielnie —
   te informacje pojawią się automatycznie w miejscu tokenów.
6. Zwróć WYŁĄCZNIE treść maila (bez tematu, bez markdown, bez cudzysłowów
   wokół całości)
7. Nigdy nie pisz rzeczy w stylu:  "Proszę o chwilę cierpliwości, a my wkrótce prześlemy szczegóły" Oferta zostanie wstawiona w tym samym momencie więc na nic nie trzeba czekać.

"""

response_agent_inquiry = Agent(name = 'response_agent',
                       description= 'Creates partial response to the clients emails',
                       model = "gemini-3.5-flash",
                       instruction=INSTRUCTION_RESPONSE_INQUIRY
)


def add_offer(node_input, ctx : Context):
    # node_input here is the models response to the clients email 
    email = node_input if isinstance(node_input, str) else str(node_input)

    if OFFER_TOKEN in email:
        # state may hold the model itself (in-memory sessions) or a plain dict (DB sessions)
        for_offer = OfferValidation.model_validate(ctx.state.get('for_offer'))
        for_offer_to_dict = [s.model_dump() for s in for_offer.selections]
        df = pd.read_parquet("catalog.parquet")
        offer = build_grouped_offer(for_offer_to_dict, df)
        before, after = email.split(OFFER_TOKEN, 1)

        final_email = llm_text_to_html(before) + offer["html"] + llm_text_to_html(after)
    else:
        final_email = llm_text_to_html(email)

    if PREVIEW_IN_BROWSER:
        _write_and_open_preview(final_email)
    # returned as this node's output so the web chat can show it to the client
    return final_email


browse_agent = Agent(name = 'browse_agent',
                     description= 'Answers to clients emails about policy of a firm', 
                     model = "gemini-3.5-flash",
                     instruction= f"""
Jesteś pracownikiem działu handlowego. Piszesz odpowiedź e-mail do klienta B2B
po polsku, w liczbie mnogiej ("przygotowaliśmy", "zapraszamy"), rzeczowo i uprzejmie.

Dane wejściowe
Mail klienta:
{{clients_email}}
Polityka firmy
{{policy_info}}

## Struktura maila (dokładnie w tej kolejności)
1. "Dzień dobry,"
2. 1–2 zdania: podziękowanie za kontakt i nawiązanie do zapytania.
3. Faktyczna odpowiedź na pytanie zadane przez klienta
4. Jedno zdanie zachęty do kontaktu, następnie "Z poważaniem," i w nowej linii "Dział Handlowy".

Jezeli klient zapyta sie o sposoby dostawy to napisz, ze w wysokim sezonie czyli sezon okołoświąteczny przyjmują zamówienia 
od 10 sztuk produktu i w takim wypadku cena dostawy jest ustalana indywidualnie. 

""")

order_agent = Agent(name = "order_agent",
                    description= 'Replies to clients emails about orders',
                    model ="gemini-3.5-flash",
                    instruction= f"""

Jesteś pracownikiem działu handlowego. Piszesz odpowiedź e-mail do klienta B2B
po polsku, w liczbie mnogiej ("przygotowaliśmy", "zapraszamy"), rzeczowo i uprzejmie.

Dane wejściowe:
Mail klienta:
{{clients_email}}
Polityka firmy
{{policy_info}}

1. "Dzień dobry,"
2. 1–2 zdania: podziękowanie za kontakt i nawiązanie do zapytania.
3. Faktyczna odpowiedź na pytanie zadane przez klienta
4. Jedno zdanie zachęty do kontaktu, następnie "Z poważaniem," i w nowej linii "Dział Handlowy".

Twoje toole:
price_order -> Funkcja zwracająca cenę za kazdy poszczególny produkt zamówiony oraz cenę zbiórczą. Skorzystaj z niego jak klient pyta się o cenę zamówienia. Nie musisz do niego dodawać zadnych parametrów. 


""" ,
tools= [price_order])


root_agent = Workflow(
    name = 'email_reponse_workflow',
    edges = [("START", normalise, extraction_agent),
             (extraction_agent, intent_router),
             (intent_router, {"order" : order_agent, 
                              "inquiry" : create_proposals,
                              "browse" : browse_agent}),
            (create_proposals, validator_agent, save_context, response_agent_inquiry, add_offer)
                              ]
)
