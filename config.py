import os
from get_items import *
from mio import load_string_list

###model mode########################################
inference_mode = False  # 推理测试的标记
inherit_optimizer = True
inherit_epoch = True
inherit_lr_scheduler = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
num_workers = 0

multi_gpu = True
if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
    multi_gpu = True
    num_workers = 2 * len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
cluster_type = 'rtx'
###model loader######################################
resume = ''

model_name = 'fusion'
n_channels = 1
n_classes = 2
use_mask_oneslice = False
use_mask = False
black_out = False
ema = False  # [0, 0.99]#False
dataloaer_settings = {'num_workers': num_workers, 'pin_memory': True}
seed = 1  # 随机种子

###dataset###########################################
# density_class_4class_64_36

data_mode = 'deepFan_Benign_and_malignant_registration_pathology'

if data_mode == 'deepFan_Benign_and_malignant_registration_pathology':
    base_list_dir = './'
    train_set_dir = base_list_dir + 'deepfan_train.csv'
    val_set_dirs = [
        base_list_dir + 'deepfan_test.csv',
    ]
    val_set_dir_inf = ''
    save_csv = [
        base_list_dir + 'deepfan_test_result.csv',
    ]
    sampler_list_dir = train_set_dir + '_weight.txt'
    n_classes = 2


##################################################################
train_set_roi_dir = None
val_set_roi_dir = None
test_set_roi_dir = None
test_set_roi_dir_1 = None

model = get_model(model_name=model_name)

###optimizer#########################################
# optimizer_name = 'Adam' #SGD Adam
# momentum =0.4
# weight_decay =1e-4

optimizer_name = 'Adam'  # SGD Adam custom custom AdamW RMSprop

if optimizer_name == 'SAM':
    optimizer_opt = {
        'base_optimizer': 'SGD',
        'momentum': 0.9,
        'weight_decay': 1e-4,
    }
if optimizer_name == 'SGD':
    optimizer_opt = {
        'momentum': 0.9,
        'weight_decay': 1e-4,
    }
if optimizer_name == 'custom_SGD':
    optimizer_opt = {
        'momentum': 0.9,
        'weight_decay': 1e-4,
    }
if optimizer_name == 'Adam':
    optimizer_opt = {
        'weight_decay': 1e-4,
        'momentum': 0.4,
    }
if optimizer_name == 'RMSprop':
    optimizer_opt = {
        'momentum': 0.0,
        'weight_decay': 0.,
        'alpha': 0.99,

    }
if optimizer_name == 'custom':
    optimizer_opt = {
        'momentum': 0.9,
        'weight_decay': 1e-4,
    }
optimizer_opt['filter_bias_and_bn'] = True

###learning rate & scheduler#########################
face_lr = False
batch_size = 2 * len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
val_batch_size = 1 * 2 * len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))  # 4* batch_size#2

start_epoch = 0
epochs = 1400
lr_controler = 11
print_freq = 10
if optimizer_name == 'Adam':
    lr = 1e-5
    epochs = 1400

###loss function#####################################
loss_function_name = "cross_entropy"  # Asl_singlelabel Asl_multilabel dice_loss cross_entropy custom focal triplet_loss_soft_margin_batch_soft
focal_gamma = 0.5
focal_alpha = None
loss_dict = None
loss_dict = {}

if loss_function_name == 'BCE':
    loss_dict = {
        'weight': [0.7, 0.3],
        'batch_size': batch_size,
    }
if loss_function_name == "class_weight_focal":
    import numpy as np

    class_weight_beta = 0.9999998
    img_num_per_cls = np.array([5342514, 352904])
    effective_num = 1.0 - np.power(class_weight_beta, img_num_per_cls)
    weights = (1.0 - class_weight_beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * int(len(img_num_per_cls))
    focal_alpha = weights
    loss_function_name = "focal"
if loss_function_name == 'triplet_loss_soft_margin_batch_soft':
    loss_dict = {
        'sample': 'weighted',
        'margin': 'soft',
    }
if loss_function_name == 'Asl_multilabel':
    loss_dict = {
        'gamma_neg': 2,
        'gamma_pos': 0,
        'clip': 0.05,
        'eps': 1e-8,  # basic ce
        'disable_torch_grad_focal_loss': True,
        'label_smooth': 0,
    }
if loss_function_name == 'Asl_singlelabel':
    loss_dict = {
        'gamma_neg': 4,
        'gamma_pos': 0,
        # 'clip': 0.05,
        'eps': 0.1,  # label smoothing
        'reduction': 'mean',
    }

loss_function = get_loss(loss_name=loss_function_name, loss_dict=loss_dict)

val_loss_dict = {}
for k, v in loss_dict.items():
    val_loss_dict[k] = v
val_loss_dict['batch_size'] = val_batch_size
val_loss_dict['disable_torch_grad_focal_loss'] = False
val_loss_function = get_loss(loss_name=loss_function_name, loss_dict=val_loss_dict)

###model saver & logg################################
if inference_mode:
    model_save_name = model_name + '_data_' + data_mode + '_inf_' + str(inference_mode) + '/' + \
                      config.resume.split('/')[-2] + '_' + config.resume.split('/')[-1].replace('.pth.tar',
                                                                                                '') + '_20241202_try_clahe'
else:
    model_save_name = model_name + '_data_' + data_mode + '_inf_' + str(
        inference_mode) + '_loss_' + loss_function_name + '_opt_' + optimizer_name + '_lr_' + str(
        lr) + '_20210220_shortexp_truewd_clahe'
mode_save_base_dir = "./" + data_mode + '/'
model_save_logg_dir = "./" + data_mode + '/' + model_save_name + '/'
model_save_freq = 30

if inference_mode:
    model_save_logg_dir = "./inf_logg/" + model_save_name + '/'

###dataloader########################################
use_sampler = True
data_source = load_string_list(train_set_dir)
print('load data source for sampler:', len(data_source))
# RandomSampler
# WeightedRandomSampler
# ct_sampler
num_samples = 6000

sampler_name = "RandomSampler"  # WeightedRandomSampler RandomSampler
sampler_setting = {}
if sampler_name == 'ct_sampler':
    sampler_setting = {
        'train_patch_path': data_source,
        'num_per_ct': 128,
        'pos_fraction': 0.5,
        'shuffle_ct': False,
    }
    sampler_setting['numct_perbatch'] = num_samples // sampler_setting['num_per_ct']
elif sampler_name == 'RandomSampler':
    sampler_setting = {
        'data_source': data_source,
        'replacement': True,
        'num_samples': num_samples
    }
elif sampler_name == 'DistributedSampler':
    sampler_setting = {
        'data_source': data_source,
        # 'replacement': True,
        # #'num_samples': None
        # 'num_samples': 450000
        # 'num_replicas': world_size,
        # 'rank': 0,
        'num_samples': num_samples}
elif sampler_name == 'WeightedRandomSampler':
    sampler_setting = {
        'sampler_list_dir': sampler_list_dir,
        'replacement': True,
        # #'num_samples': None
        # 'num_samples': 450000
        # 'num_replicas': world_size,
        # 'rank': 0,
        'num_samples': num_samples}
sampler_setting_2record = {k: v for k, v in sampler_setting.items() if k != 'data_source'}
if use_sampler:
    sampler = get_sampler(sampler_name=sampler_name, sampler_setting=sampler_setting)

##lr scheduler######################################
use_lr_scheduler = False

lr_scheduler_name = 'MultiStepLR'  # MultiStepLR OneCycleLR

hyper_parameters = {
    'alpha': 0.0,
    'beta': 0.4,
    'lambda': 3000,
    'lr': lr
}

if lr_scheduler_name == 'MultiStepLR':
    lr_scheduler_opt = {
        # 'milestones': [60-64, 90-64],
        # 'milestones': [25, 50, 75],
        # 'milestones': [30, 60, 90],
        'milestones': [300, 600, 900],
        # 'milestones': [50, 100, 150, 200],
        # 'milestones': [120, 180, 240],
        # 'milestones': [60, 120, 180],
        'gamma': 0.1,

    }
elif lr_scheduler_name == 'CosineAnnealingWarmRestarts':
    lr_scheduler_opt = {
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 0.0001,
        'last_epoch': -1,

    }
elif lr_scheduler_name == 'OneCycleLR':
    lr_scheduler_opt = {
        'max_lr': lr,
        'steps_per_epoch': num_samples // batch_size,
        'epochs': 160,
        'pct_start': 0.2,
    }

###data aug #########################################


# nodule
pre_crop = False  # (110, 110, 110)
center_crop = (128, 128, 128)
final_size = (128, 128, 128)
train_crop = (128, 128, 128)
rotation = 'big'
ran_zoom = [0.85, 1.15]
random_crop = [6, 6, 6]
normalize = True
random_brightness_limit = [-0.1, 0.1]
random_contrast_limit = [-0.1, 0.1]
random_gamma_limit = False  # [90, 110]
offset_max = False
confusion = False
label_pos = [1]

shear = False
flip = True
scale = 1 / 255.0
pad = False  # [(20, 20), (20, 20), (20, 20)]
test_zoom = False  # (1.1, 1.1, 1)
gpu_aug = True
clahe = False  # [1.0, (30, 30)] #clipLimit tileGridSize

confusion = False


train_dataaug_opt = dict(
    center_crop=False,
    scale=scale,
    label_pos=label_pos,
    final_size=final_size,
    rotation=rotation,
    shear=shear,
    train_crop=train_crop,
    random_crop=random_crop,
    flip=flip,
    offset_max=offset_max,
    ran_zoom=ran_zoom,
    train_flag='train',
    pad=pad,
    normalize=normalize,
    test_zoom=False,
    use_mask=use_mask,
    pre_crop=pre_crop,
    gpu_aug=gpu_aug,
    random_brightness_limit=random_brightness_limit,
    random_contrast_limit=random_contrast_limit,
    random_gamma_limit=random_gamma_limit,
    black_out=black_out,
    confusion=confusion,
    black_in=False,
    new_black_out=False,
    new_black_in=False,
    TBMSL_NET_opt=False,
    use_mask_oneslice=use_mask_oneslice,
    clahe=clahe,
)

val_dataaug_opt = dict(
    center_crop=center_crop,
    scale=scale,
    label_pos=label_pos,
    final_size=final_size,
    rotation=False,
    shear=False,
    train_crop=False,
    random_crop=False,
    flip=False,
    offset_max=False,
    ran_zoom=False,
    train_flag='val',
    pad=pad,
    normalize=normalize,
    test_zoom=test_zoom,
    use_mask=use_mask,
    pre_crop=pre_crop,
    random_brightness_limit=False,
    random_contrast_limit=False,
    random_gamma_limit=False,
    black_out=False,
    black_in=False,
    new_black_out=False,
    new_black_in=False,
    TBMSL_NET_opt=False,
    use_mask_oneslice=use_mask_oneslice,
    clahe=clahe,
)
val_dataaug_opts = [val_dataaug_opt] * len(val_set_dirs)

###parameters for logg################################
model_mode_record = dict(
    Inference_Mode=inference_mode,
    GPU_Used=os.environ['CUDA_VISIBLE_DEVICES'],
    Inherit_optimizer=inherit_optimizer,
    Inherit_epoch=inherit_epoch,
    Inherit_lr_scheduler=inherit_lr_scheduler,
)
model_load_record = dict(
    Model=model_name,
    Input_Channel=n_channels,
    Output_Classes=n_classes,
    Resume=resume,
    Seed=seed,
)
model_save_record = dict(
    Model_Save_Dir=mode_save_base_dir,
    Model_Logg_Dir=model_save_logg_dir,
    Ema=ema,
)
dataset_record = dict(
    Train_Set_Dir=train_set_dir,
    val_set_dirs=val_set_dirs,
    Train_Set_Roi_Dir=train_set_roi_dir,
    Val_Set_Roi_Dir=val_set_roi_dir,
)
optimizer_record = dict(
    Optimizer_Name=optimizer_name,
    Optimizer_opt=optimizer_opt,
)
lr_scheduler_record = dict(
    Face_Learning_Rate_Scheduler=face_lr,
    Batch_Size=batch_size,
    Start_Epoch=start_epoch,
    End_Epoch=epochs,
    Learning_Rate=lr,
    Lr_Controler=lr_controler,
    hyper_para=hyper_parameters,
    use_lr_scheduler=use_lr_scheduler,
    lr_scheduler_name=lr_scheduler_name,
    lr_scheduler_opt=lr_scheduler_opt,
)

loss_record = dict(
    Loss_Func=loss_function_name,
    Loss_dict=loss_dict,
    Confusion=confusion,
    Val_Loss_Func=val_loss_dict,
)

dataloader_record = dict(
    Use_Sampler=use_sampler,
    Sampler_Name=sampler_name,
    Sampler_List_Dir=sampler_list_dir,
    sampler_setting=sampler_setting_2record,
    batch_size=batch_size,
    val_batch_size=val_batch_size,
)
