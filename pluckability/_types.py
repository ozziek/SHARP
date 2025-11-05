from typing import TypedDict


class SHARPCard(TypedDict):
    id: int  # maps to the flashcard id in our dataset
    annotation_id: int  # maps to the annotation id in our dataset
    source_url: str
    source_meta: dict
    highlight: str
    highlight_interpretation: str
    content: str
    pluckable: bool
    tags: list[str]
