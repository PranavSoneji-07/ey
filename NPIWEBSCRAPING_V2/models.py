from pydantic import BaseModel
from typing import List, Optional

class ScrapedProviderData(BaseModel):
    name: Optional[str] = None
    profession: Optional[str] = None
    taxonomies: List[str] = []
    addresses: List[str] = []
    phone_numbers: List[str] = []
    source_urls: List[str] = []
