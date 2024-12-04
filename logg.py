#import tensorflow as tf
import numpy as np
import scipy.misc
import os
import logging
import config

def get_logger(logg_dir):
    logger = logging.getLogger('LungSeg_Class')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('{}training.log'.format(logg_dir, ))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def logg_init(logg_dir = config.model_save_logg_dir):
    logger = get_logger(config.model_save_logg_dir)
    logger.info('Model Name:' + config.model_save_name)
        
    logger.info('Model Mode Settings: %s' % str(config.model_mode_record))
    logger.info('Model Load Settings: %s' % str(config.model_load_record))
    logger.info('Model Save Settings: %s' % str(config.model_save_record))
    logger.info('Dataset Settings: %s'    % str(config.dataset_record))
    logger.info('Dataloader Settings: %s' % str(config.dataloader_record))
    #print str(config.lr_scheduler_record)
    logger.info('Learning Rate & Scheduler Settings: %s' % str(config.lr_scheduler_record))
    logger.info('Optimizer Settings: %s' % str(config.optimizer_record))
    logger.info('Loss Function Settings: %s' % str(config.loss_record))
    logger.info('Train Data Aug Settings: %s' % str(config.train_dataaug_opt))
    logger.info('Val Data Aug Settings: %s' % str(config.val_dataaug_opt))
    #logger.info('Test Data Aug Settings: %s' % str(config.test_dataaug_opt))
    return logger

def logg_init_new(logg_dir,
              model_save_name,
              model_mode_record,
              model_load_record,
              model_save_record,
              dataset_record,
              dataloader_record,
              lr_scheduler_record,
              optimizer_record,
              loss_record,
              train_dataaug_opt,
              val_dataaug_opts):
    logger = get_logger(logg_dir)
    logger.info('Model Name:' + model_save_name)

    logger.info('Model Mode Settings: %s' % str(model_mode_record))
    logger.info('Model Load Settings: %s' % str(model_load_record))
    logger.info('Model Save Settings: %s' % str(model_save_record))
    logger.info('Dataset Settings: %s' % str(dataset_record))
    logger.info('Dataloader Settings: %s' % str(dataloader_record))
    # print str(config.lr_scheduler_record)
    logger.info('Learning Rate & Scheduler Settings: %s' % str(lr_scheduler_record))
    logger.info('Optimizer Settings: %s' % str(optimizer_record))
    logger.info('Loss Function Settings: %s' % str(loss_record))
    logger.info('Train Data Aug Settings: %s' % str(train_dataaug_opt))
    logger.info('Val Data Aug Settings: %s' % str(val_dataaug_opts))
    # logger.info('Test Data Aug Settings: %s' % str(config.test_dataaug_opt))
    return logger
