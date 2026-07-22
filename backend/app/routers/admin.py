from fastapi import APIRouter, Depends, HTTPException
from rq.job import Job
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_session
from app.queue import get_queue
from app.schemas import ModelVersionOut
from app.services.recommender import get_active_model

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/models/current", response_model=ModelVersionOut)
async def current_model(session: AsyncSession = Depends(get_session)):
    model = await get_active_model(session)
    if model is None:
        raise HTTPException(status_code=404, detail="Активная модель не найдена")
    return model


@router.post("/retrain", status_code=202)
async def retrain(simulate_degraded: bool = False):
    job = get_queue().enqueue(
        "app.tasks.retrain", simulate_degraded=simulate_degraded, job_timeout=1800
    )
    return {"status": "queued", "job_id": job.id}


@router.post("/import", status_code=202)
async def import_data():
    job = get_queue().enqueue("app.tasks.import_demo_data", job_timeout=600)
    return {"status": "queued", "job_id": job.id}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=get_queue().connection)
    except Exception:
        raise HTTPException(status_code=404, detail="Задача не найдена") from None
    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.return_value(),
        "error": (job.exc_info or "").splitlines()[-1] if job.exc_info else None,
    }
