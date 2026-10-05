# Here i will have all the necessary classes for the Workflow 

from typing import Optional, Literal, List
from pydantic import BaseModel, Field


class Product(BaseModel):
    url: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[int] = None
    alcohol: Literal["any", "none", "required", "specific"] = "any"
    alcohol_detail: Optional[str] = None
    price_min: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Lower bound of the client's budget. Leave null if no lower bound is stated. "
            f""
            "Never set price_min equal to price_max."
        ),
    )
    price_max: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Upper bound of the client's budget. If the client gives a single price "
            "('around 100 zł', 'up to 100 zł'), put it here and leave price_min null."
        ),
    )
    price_basis: Literal["netto", "brutto"] = "brutto"
    price_per: Literal["piece", "person", "total"] = "piece"
    price_includes_delivery: bool = False
    delivery: Literal[
        "to_addresses", "pickup", "to_client_warehouse", "unspecified"
    ] = "unspecified"

    product_id: Optional[int] = None
    code: Optional[str] = Field(None,
    description=("If you see any codes like PL44, PL121 put them into code"),)


class EmailExtraction(BaseModel):
    products: List[Product] = Field(default_factory=list)
    deadline: Optional[str] = None  # "YYYY-MM-DD" or None
    policy_questions: List[
        Literal[
            "discount", "delivery_time", "different_addreses",
            "personalization", "greeting_card", "international_delivery", "payment"
        ]
    ] = Field(default_factory=list)
    intent: Literal["order", "inquiry", "browse"] = "inquiry"

class GroupSelection(BaseModel):
    label : str
    group_index: int
    quantity : int
    codes: List[str] = Field(default_factory=list)

class OfferValidation(BaseModel):
    selections: List[GroupSelection]
    reasoning: str
    note: str

