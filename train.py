import sys
from collections import OrderedDict
from dataloader import create_dataloader
from options.base_options import Options
from util.iter_counter import EpochCounter

opt = Options()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = create_dataloader(opt)

# create trainer for our model
# trainer = Pix2PixTrainer(opt)
#
# create tool for counting iterations
epoch_counter = EpochCounter(opt)
#
# # create tool for visualization
# visualizer = Visualizer(opt)

for epoch in epoch_counter.training_epochs():
    epoch_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader):
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

    trainer.update_learning_rate(epoch)

    epoch_counter.record_epoch_end()
    trainer.save('latest')
    trainer.save(epoch)

print('Training was successfully finished.')
