import glob
import re
import numpy as np
import sys
import os

out_dirs = os.listdir('output')
for d in out_dirs:
    log_files = glob.glob('output/{}/202*.log'.format(d))
    latest_log_file = sorted(log_files, key=os.path.getmtime)[-1]
    with open(latest_log_file, 'r') as f:
        lines = f.readlines()
    print(d)
    for i in range(len(lines)):
        # if '|  aAcc |  mIoU |  mAcc |' in lines[i] or '|  aAcc | mIoU | mAcc |' in lines[i]:
        if 'aAcc' in lines[i] and 'mIoU' in lines[i] and '|' in lines[i]:
            print(lines[i+2].strip())


#
# for root_dir in root_dirs:
#     fs =
#     log_files += fs
#
# for file_path in log_files:
#     skip = False
#     if excluded is not None:
#         for ex in excluded:
#             if ex in file_path:
#                 skip = True
#                 break
#     if skip:
#         continue
#     top1_list = []
#     top5_list = []
#     loss_list = []
#     baseline_speed = 0
#     exp_speed = 0
#     with open(file_path, 'r') as f:
#         origin_lines = f.readlines()
#         for l in origin_lines:
#             if 'baseline speed' in l:
#                 baseline_speed = get_value_by_pattern(speed_pattern, l)
#             elif 'bbf speed' in l or 'exp speed' in l or 'ent speed' in l:
#                 exp_speed = get_value_by_pattern(speed_pattern, l)
#                 break
#
#         log_lines = [l for l in origin_lines if 'parallel' not in l and 'top1' in l and 'top5' in l and 'loss' in l and 'beginning' not in l]
#         avg_loss = '----'
#         params = '----'
#         train_speed = '----'
#         deploy_speed = '----'
#         for l in origin_lines[-5:]:
#             if 'TRAIN LOSS collected over last' in l:
#                 avg_loss = l.strip()[-8:]
#             if 'num of params in hdf5' in l:
#                 params = l.strip().split('=')[1]
#             if 'TRAIN speed' in l:
#                 train_speed = float(l.strip().split('=')[-1])
#                 train_speed = '{:.2f}'.format(train_speed)
#             if 'DEPLOY TEST' in l:
#                 ll = l.strip().split(' ')
#                 examples = int(ll[4])
#                 secs = float(ll[6])
#                 deploy_speed = examples / secs
#                 deploy_speed = '{:.2f}'.format(deploy_speed)
#         last_lines = log_lines[-num_logs:]
#     for l in last_lines:
#         if 'top1' not in l or 'loss' not in l or 'top5' not in l:
#             continue
#         top1, top5, loss = parse_top1_top5_loss_from_log_line(l)
#         top1_list.append(top1)
#         top5_list.append(top5)
#         loss_list.append(loss)
#     if len(top1_list) < num_logs:
#         continue
#     # network_try_arg = file_path.split('/')[1].replace('_train', '')
#     network_try_arg = file_path.replace('_train/log.txt', '')
#     last_validation = last_lines[-1]
#     last_epoch_pattern = re.compile('epoch (\d+)')
#
#     last_epoch = int(last_epoch_pattern.findall(last_validation)[0])
#
#     if exp_speed > 0:
#         speedup = exp_speed / baseline_speed
#     else:
#         speedup = 0
#
#     thresh = ''
#     flops_r = ''
#     for ol in origin_lines[-70:]:
#         if 'thres 1e-05' in ol:
#             thresh = '1e-5'
#         elif 'thres 1e-06' in ol:
#             thresh = '1e-6'
#         if 'FLOPs' in ol:
#             flops_r = ol[ol.index('FLOPs'):].strip()
#             # print('----xxxxxxxxxxx')
#             # print(flops_r)
#             # exit(0)
#
#     msg = '{} \t maxtop1={:.3f}, spdup={:.3f}, mean={:.3f}+-{:.3f}, loss={:.5f}, {} logs, tr_loss={}, para={}, ts={}, ds={}, last={}'.format(network_try_arg,
#             np.max(top1_list), speedup, np.mean(top1_list), np.std(top1_list), np.mean(loss_list),
#              len(top1_list), avg_loss, params, train_speed, deploy_speed, last_epoch)
#     if len(flops_r) > 0:
#         msg += '  ' + thresh + ':' + flops_r
#     print(msg)
