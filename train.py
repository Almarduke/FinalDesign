import sys
import time
from collections import OrderedDict
from dataloader import create_dataloader
from options.base_options import Options
from trainers.train_manager import TrainManager
from util.epoch_counter import EpochCounter
from util.visualizer import Visualizer

opt = Options()

# load the dataset
dataloader = create_dataloader(opt)

# create trainer for our model
trainer = TrainManager(opt)
#
# create tool for counting iterations
epoch_counter = EpochCounter(opt)
#
# create tool for visualization
visualizer = Visualizer(opt)

for epoch in epoch_counter.training_epochs():
    epoch_counter.record_epoch_start(epoch)
    for batch_id, (real_imgs, labels) in enumerate(dataloader):
        iter_start_time = time.time()

        data_i = (real_imgs, labels)
        trainer.run_generator_one_step(data_i)
        trainer.run_discriminator_one_step(data_i)

        running_time = time.time() - iter_start_time
        visualizer.print_current_errors(epoch, batch_id, running_time, losses)

        if batch_id % 200 == 0:
            generated_imgs = trainer.get_latest_generated()
            visualizer.save_images(epoch, batch_id, labels, real_imgs, generated_imgs)

    epoch_counter.record_epoch_end()
    trainer.save(epoch)
    trainer.update_learning_rate(epoch)

print('Training was successfully finished.', flush=True)
