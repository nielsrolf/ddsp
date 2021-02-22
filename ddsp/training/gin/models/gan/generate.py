import os


options = {
    'base': 'models/gan/base.gin',
    'models': [
        {'models/gan/lsgan.gin': [
            'models/gan/conv_discriminator.gin',
            'models/gan/mfcc_discriminator.gin'
        ]},
        {'models/gan/wgan.gin': [
            'models/gan/conv_discriminator.gin',
            'models/gan/mfcc_discriminator.gin'
        ]},
        'models/gan/ae.gin'
    ],
    'decoder': [
        'models/gan/conv_decoder.gin',
        'models/gan/rnn_ddsp_decoder.gin'
    ]
}

commands_file = "run_all.sh"
filename_prefix = "generated/exp01_"

base_command = """
ddsp_run \\
    --mode=train \\
    --alsologtostderr \\
    --gin_file=datasets/tfrecord.gin \\
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \\
    --gin_param="batch_size=8" \\
    --gin_param="train_util.train.num_steps=36000" \\
    --gin_param="train_util.train.steps_per_save=300" \\
    --gin_param="train_util.train.steps_per_summary=100" \\
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \\
"""


def command_with_ginfile(filepath):
    save_dir = filepath.split("/")[-1][:-4]
    filepath = os.path.abspath(filepath)
    return (base_command + \
        f"    --save_dir=\"$RESULTS_DIR/{save_dir}\" \\\n"
        f"    --gin_file={filepath} \n\n")


def combine_all(options):
    if isinstance(options, str):
        yield [options]
        return
    if isinstance(options, list):
        for value in options:
            for expanded in combine_all(value):
                yield expanded
        return
    options = dict(**options)
    keys = list(options.keys())
    head = keys[0]
    # if the current key is a gin file, include that
    if head.endswith(".gin"):
        first_include = [head]
    else:
        first_include = []
    # if there is only one key in the dict, iterate over its val
    if len(keys) == 1:
        for  val in combine_all(options[head]):
            yield first_include + val
        return
    # there is a tail and we combine each combination for
    # head with each combination for tail
    tail = keys[1:]
    for head_val in combine_all(options.pop(head)):
        for tail_val in combine_all(options):
            yield first_include + head_val + tail_val


commands = []

gin_include = "include '{}'\n".format
for combination in combine_all(options):
    filename = filename_prefix + "_".join([i.split("/")[-1][:-4] for i in combination]) + ".gin"
    commands += [command_with_ginfile(filename)]
    with open(filename, "w") as f:
        for include_file in combination:
            f.write(gin_include(include_file))

with open(commands_file, "w") as f:
    for cmd in commands:
        f.writelines(cmd)
