import asyncio
from .synthesis_service import run_synthesis

def run_synthesis_task(args, data_dir, X_num_raw, X_cat_raw, confirmed_domain_data, confirmed_info_data, job_id, results_dict):
    """
    This function is intended to be run in a separate process.
    It runs the synthesis and puts the result in a shared dictionary.
    """
    async def main():
        # Simple progress reporter that writes into the shared dict
        def progress_report(payload: dict):
            results_dict[job_id] = {**results_dict.get(job_id, {}), **payload}
        try:
            synthesized_csv_path, _ = await run_synthesis(
                args=args,
                data_dir=data_dir,
                X_num_raw=X_num_raw,
                X_cat_raw=X_cat_raw,
                confirmed_domain_data=confirmed_domain_data,
                confirmed_info_data=confirmed_info_data,
                progress_report=progress_report,
            )
            results_dict[job_id] = {**results_dict.get(job_id, {}), "status": "completed", "path": synthesized_csv_path}
        except Exception as e:
            results_dict[job_id] = {**results_dict.get(job_id, {}), "status": "failed", "error": str(e)}

    asyncio.run(main())
