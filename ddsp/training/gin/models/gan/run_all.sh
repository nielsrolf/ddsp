
ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_lsgan_conv_discriminator_conv_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_lsgan_conv_discriminator_conv_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_lsgan_conv_discriminator_rnn_ddsp_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_lsgan_conv_discriminator_rnn_ddsp_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_lsgan_mfcc_discriminator_conv_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_lsgan_mfcc_discriminator_conv_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_lsgan_mfcc_discriminator_rnn_ddsp_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_lsgan_mfcc_discriminator_rnn_ddsp_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_wgan_conv_discriminator_conv_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_wgan_conv_discriminator_conv_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_wgan_conv_discriminator_rnn_ddsp_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_wgan_conv_discriminator_rnn_ddsp_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_wgan_mfcc_discriminator_conv_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_wgan_mfcc_discriminator_conv_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_wgan_mfcc_discriminator_rnn_ddsp_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_wgan_mfcc_discriminator_rnn_ddsp_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_ae_conv_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_ae_conv_decoder.gin 


ddsp_run \
    --mode=train \
    --alsologtostderr \
    --gin_file=datasets/tfrecord.gin \
    --gin_param="TFRecordProvider.file_pattern='$URMP_MONO'" \
    --gin_param="batch_size=8" \
    --gin_param="train_util.train.num_steps=36000" \
    --gin_param="train_util.train.steps_per_save=300" \
    --gin_param="train_util.train.steps_per_summary=100" \
    --gin_param="trainers.Trainer.checkpoints_to_keep=2" \
    --save_dir="$RESULTS_DIR/exp01_base_ae_rnn_ddsp_decoder" \
    --gin_file=/mnt/raid/ni/niels/ddsp/ddsp/training/gin/models/gan/generated/exp01_base_ae_rnn_ddsp_decoder.gin 

