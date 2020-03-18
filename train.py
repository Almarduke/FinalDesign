import sys
import time
from collections import OrderedDict
from dataloader import create_dataloader
from options.base_options import Options
from trainers.pix2pix_trainer import Pix2PixTrainer
from util.iter_counter import EpochCounter
from util.visualizer import Visualizer

opt = Options()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)
#
# create tool for counting iterations
epoch_counter = EpochCounter(opt)
#
# create tool for visualization
visualizer = Visualizer(opt)

for epoch in epoch_counter.training_epochs():
    epoch_counter.record_epoch_start(epoch)
    for batch_id, data_i in enumerate(dataloader):
        iter_start_time = time.time()

        # Training
        # train generator
        if batch_id % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        running_time = time.time() - iter_start_time
        visualizer.print_current_errors(epoch, batch_id, running_time, losses)

    trainer.update_learning_rate(epoch)

    epoch_counter.record_epoch_end()
    trainer.save('latest')
    trainer.save(epoch)

print('Training was successfully finished.')
