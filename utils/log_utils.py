import os
import csv

CSV_DATA_MISMATCH = "Row data length must match the file header length"


def write_results_csv(file_name, headers_name, row_data, operation="a"):
    if len(headers_name) != len(row_data):
        raise ValueError(CSV_DATA_MISMATCH)
    _write_data = list()

    if not os.path.exists(file_name):
        operation = "w"
        _write_data.append(headers_name)

    _write_data.append(row_data)

    with open(file_name, operation) as f:
        writer = csv.writer(f)
        _ = [writer.writerow(i) for i in _write_data]


def file_is_empty(path):
    return os.stat(path).st_size == 0


def save_to_file(path, dict_saver):
    header = list(dict_saver.keys())
    values = list(dict_saver.values())
    write_results_csv(path, header, values)


def log_values(
    cost,
    grad_norms,
    epoch,
    batch_id,
    step,
    log_likelihood,
    reinforce_loss,
    bl_loss,
    tb_logger,
    opts,
):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(
        "epoch: {}, train_batch_id: {}, avg_cost: {}".format(epoch, batch_id, avg_cost)
    )

    print("grad_norm: {}, clipped: {}".format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value("avg_cost", avg_cost, step)

        tb_logger.log_value("actor_loss", reinforce_loss.item(), step)
        tb_logger.log_value("nll", -log_likelihood.mean().item(), step)

        tb_logger.log_value("grad_norm", grad_norms[0], step)
        tb_logger.log_value("grad_norm_clipped", grad_norms_clipped[0], step)

        if opts.baseline == "critic":
            tb_logger.log_value("critic_loss", bl_loss.item(), step)
            tb_logger.log_value("critic_grad_norm", grad_norms[1], step)
            tb_logger.log_value("critic_grad_norm_clipped", grad_norms_clipped[1], step)
