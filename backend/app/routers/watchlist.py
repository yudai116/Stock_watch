import re
from fastapi import APIRouter, HTTPException
from app.models import WatchlistResponse, AddTickerRequest
from app.storage import watchlist_store
from app.services import market_data

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,2})?$")


@router.get("", response_model=WatchlistResponse)
def get_watchlist():
    return WatchlistResponse(tickers=watchlist_store.load())


@router.post("", response_model=WatchlistResponse)
def add_ticker(body: AddTickerRequest):
    ticker = body.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=422, detail=f"Invalid ticker format: {ticker}")
    valid, name, mkt = market_data.validate_ticker(ticker)
    if not valid:
        raise HTTPException(status_code=404, detail=f"Symbol {ticker} not found on Yahoo Finance")
    market_data.invalidate_cache(ticker)
    tickers = watchlist_store.add(ticker)
    return WatchlistResponse(tickers=tickers)


@router.delete("/{ticker}", response_model=WatchlistResponse)
def remove_ticker(ticker: str):
    tickers = watchlist_store.remove(ticker.upper())
    return WatchlistResponse(tickers=tickers)
