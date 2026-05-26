from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, HTTPException
from app.models import StockScoresResponse, StockScore, StockDetail, ValidateResponse
from app.services import market_data

router = APIRouter()
_executor = ThreadPoolExecutor(max_workers=10)


@router.get("/scores", response_model=StockScoresResponse)
def get_scores(tickers: str):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:50]
    results: list[StockScore] = []
    errors: dict[str, str] = {}

    futures = {_executor.submit(market_data.fetch_score, t): t for t in ticker_list}
    for future in as_completed(futures):
        t = futures[future]
        try:
            results.append(future.result())
        except Exception as e:
            errors[t] = str(e)

    results.sort(key=lambda x: x.score, reverse=True)
    return StockScoresResponse(results=results, errors=errors)


@router.get("/{ticker}/detail", response_model=StockDetail)
def get_detail(ticker: str):
    try:
        return market_data.fetch_detail(ticker.upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/validate", response_model=ValidateResponse)
def validate(ticker: str):
    valid, name, mkt = market_data.validate_ticker(ticker.upper())
    if not valid:
        return ValidateResponse(valid=False, error="Symbol not found")
    return ValidateResponse(valid=True, name=name, market=mkt)
