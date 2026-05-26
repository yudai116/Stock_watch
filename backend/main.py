import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from app.config import settings
from app.routers import stocks, watchlist
from app.services.market_data import refresh_all_cached, cache_size
from app.models import HealthResponse

_start_time = time.time()

app = FastAPI(title="Stock Watch API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_allow_all else settings.cors_origins,
    allow_credentials=not settings.cors_allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(watchlist.router, prefix="/api/watchlist", tags=["watchlist"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])

_scheduler = BackgroundScheduler()


@app.on_event("startup")
def startup():
    _scheduler.add_job(refresh_all_cached, "interval", minutes=4, id="cache_refresh")
    _scheduler.start()


@app.on_event("shutdown")
def shutdown():
    _scheduler.shutdown(wait=False)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        cache_size=cache_size(),
        uptime_seconds=round(time.time() - _start_time, 1),
    )
