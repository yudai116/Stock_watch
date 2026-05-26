from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ScoreComponent(BaseModel):
    score: int
    max: int
    value: Optional[float] = None
    signal: str


class StockScore(BaseModel):
    ticker: str
    market: str
    name: str
    price: Optional[float]
    currency: str
    change_pct: Optional[float]
    score: int
    score_components: Dict[str, ScoreComponent]
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    pe_label: Optional[str]
    last_updated: str


class StockScoresResponse(BaseModel):
    results: List[StockScore]
    errors: Dict[str, str]


class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class SeriesPoint(BaseModel):
    date: str
    value: float


class MACDPoint(BaseModel):
    date: str
    macd: float
    signal: float
    histogram: float


class BBPoint(BaseModel):
    date: str
    upper: float
    middle: float
    lower: float
    pct_b: float


class StockDetail(BaseModel):
    ticker: str
    market: str
    name: str
    price: Optional[float]
    currency: str
    change_pct: Optional[float]
    score: int
    score_components: Dict[str, ScoreComponent]
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    pe_label: Optional[str]
    history: List[OHLCVPoint]
    rsi_series: List[SeriesPoint]
    macd_series: List[MACDPoint]
    bb_series: List[BBPoint]
    ma20_series: List[SeriesPoint]
    ma50_series: List[SeriesPoint]
    last_updated: str


class WatchlistResponse(BaseModel):
    tickers: List[str]


class AddTickerRequest(BaseModel):
    ticker: str


class ValidateResponse(BaseModel):
    valid: bool
    name: Optional[str] = None
    market: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    cache_size: int
    uptime_seconds: float
