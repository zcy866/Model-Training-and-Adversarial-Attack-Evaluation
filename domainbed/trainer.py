import collections
import json
import time
import copy
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.data

from domainbed.dataset_utils import build_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed import swad as swad_module

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(args, hparams, n_steps, checkpoint_freq, logger):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    data_info, dataset_train, dataset_val = build_dataset(args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=hparams["test_batchsize"],
        num_workers=hparams["num_workers"],
        drop_last=False
    )

    #######################################################
    # setup algorithm (model)
    #######################################################
    hparams["steps"] = args.steps
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        data_info.num_classes,
        hparams
    )

    algorithm.to(device)

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        data_loader_val,
        logger,
        evalmode=args.evalmode,
        debug=args.debug
    )

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, hparams["swad"])
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    data_loader_iter = iter(data_loader_train)
    best_accuracy = 0
    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]

        try:
            batches = next(data_loader_iter)
        except:
            data_loader_iter = iter(data_loader_train)
            batches = next(data_loader_iter)
        # to device
        batches = {
            key: tensorlist.to(device) for key, tensorlist in batches.items()
        }
        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + list(results.keys())
            # merge results
            results.update(summaries)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.model_save and step >= args.model_save:
                save_path = os.path.join('./save_model', args.task, args.algorithm, args.name, str(step))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "model": algorithm.cpu().state_dict()
                }
                algorithm.to(device)
                if not args.debug:
                    torch.save(save_dict, os.path.join('./save_model', args.task, args.algorithm, args.name, str(step), "model.pkl"))
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % save_path)
                if best_accuracy < summaries["val_accuracy"]:
                    best_accuracy = summaries["val_accuracy"]
                    save_path = os.path.join('./save_model', args.task, args.algorithm, args.name, "best_model")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(save_dict, os.path.join('./save_model', args.task, args.algorithm, args.name, "best_model", "model.pkl"))

            # swad
            if swad:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["val_accuracy"], results["val_accuracy"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset

    logger.info(f"best_accuracy = {best_accuracy:.3%}")
    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(data_loader_train, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)
        save_path = os.path.join('./save_model', args.task, args.algorithm, args.name, "swad")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dict = {
            "args": vars(args),
            "model_hparams": dict(hparams),
            "model": swad_algorithm.module.cpu().state_dict()
        }
        algorithm.to(device)
        if not args.debug:
            torch.save(save_dict, os.path.join('./save_model', args.task, args.algorithm, args.name, "swad", "model.pkl"))
        else:
            logger.debug("DEBUG Mode -> no save (org path: %s)" % save_path)
