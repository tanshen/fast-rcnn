#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib
matplotlib.use('Agg')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import os.path

CLASSES = ('__background__','whale_48813','whale_09913','whale_45062','whale_74162','whale_99558' ,'whale_59255' ,'whale_87291' ,'whale_33152' ,'whale_88147' ,'whale_77693' ,'whale_74625' ,'whale_06069' ,'whale_62939' ,'whale_67036' ,'whale_78785' ,'whale_75682' ,'whale_08017' ,'whale_23574' ,'whale_46169' ,'whale_11242' ,'whale_03227' ,'whale_24815' ,'whale_36231' ,'whale_89615' ,'whale_64989' ,'whale_52342' ,'whale_63948' ,'whale_80947' ,'whale_80247' ,'whale_11076' ,'whale_70138' ,'whale_07863' ,'whale_36300' ,'whale_64634' ,'whale_22448' ,'whale_68116' ,'whale_30331' ,'whale_64006' ,'whale_54796' ,'whale_48415' ,'whale_09181' ,'whale_86041' ,'whale_81818' ,'whale_99573' ,'whale_47734' ,'whale_58010' ,'whale_23525' ,'whale_48490' ,'whale_02411' ,'whale_08439' ,'whale_08181' ,'whale_78280' ,'whale_23847' ,'whale_29569' ,'whale_67685' ,'whale_49530' ,'whale_48386' ,'whale_49491' ,'whale_39915' ,'whale_38008' ,'whale_50021' ,'whale_81875' ,'whale_10977' ,'whale_82548' ,'whale_36154' ,'whale_85670' ,'whale_23367' ,'whale_98645' ,'whale_87155' ,'whale_59173' ,'whale_74683' ,'whale_76782' ,'whale_73592' ,'whale_00195' ,'whale_43971' ,'whale_79439' ,'whale_53079' ,'whale_52505' ,'whale_97882' ,'whale_95370' ,'whale_94176' ,'whale_63541' ,'whale_70904' ,'whale_28892' ,'whale_16738' ,'whale_05487' ,'whale_83892' ,'whale_57251' ,'whale_72820' ,'whale_98996' ,'whale_12820' ,'whale_66852' ,'whale_27085' ,'whale_90820' ,'whale_23821' ,'whale_06339' ,'whale_83791' ,'whale_37154' ,'whale_87622' ,'whale_40169' ,'whale_15079' ,'whale_52749' ,'whale_11708' ,'whale_82064' ,'whale_60729' ,'whale_17604' ,'whale_28263' ,'whale_19906' ,'whale_34656' ,'whale_61461' ,'whale_96385' ,'whale_73684' ,'whale_23118' ,'whale_87604' ,'whale_37654' ,'whale_39689' ,'whale_26288' ,'whale_37658' ,'whale_55550' ,'whale_69459' ,'whale_90929' ,'whale_35594' ,'whale_41125' ,'whale_79823' ,'whale_46974' ,'whale_35844' ,'whale_03623' ,'whale_18989' ,'whale_47700' ,'whale_37269' ,'whale_85464' ,'whale_53580' ,'whale_38681' ,'whale_54574' ,'whale_44127' ,'whale_73080' ,'whale_61260' ,'whale_48966' ,'whale_64714' ,'whale_55079' ,'whale_98939' ,'whale_26397' ,'whale_91826' ,'whale_64496' ,'whale_17785' ,'whale_73136' ,'whale_49237' ,'whale_68789' ,'whale_20448' ,'whale_31594' ,'whale_27820' ,'whale_19041' ,'whale_06334' ,'whale_32087' ,'whale_82089' ,'whale_74439' ,'whale_60921' ,'whale_65057' ,'whale_75932' ,'whale_97542' ,'whale_02839' ,'whale_36851' ,'whale_25878' ,'whale_15434' ,'whale_89211' ,'whale_09454' ,'whale_29172' ,'whale_08637' ,'whale_35426' ,'whale_86377' ,'whale_51538' ,'whale_66711' ,'whale_80405' ,'whale_05349' ,'whale_97688' ,'whale_09651' ,'whale_26686' ,'whale_54920' ,'whale_90271' ,'whale_36437' ,'whale_65586' ,'whale_33723' ,'whale_79648' ,'whale_09062' ,'whale_59627' ,'whale_97924' ,'whale_38817' ,'whale_05661' ,'whale_11555' ,'whale_13288' ,'whale_49135' ,'whale_92465' ,'whale_90957' ,'whale_12074' ,'whale_34488' ,'whale_79166' ,'whale_33140' ,'whale_86206' ,'whale_82554' ,'whale_26654' ,'whale_22297' ,'whale_95831' ,'whale_15519' ,'whale_38288' ,'whale_72327' ,'whale_90911' ,'whale_10005' ,'whale_38894' ,'whale_03103' ,'whale_81136' ,'whale_84178' ,'whale_95091' ,'whale_39293' ,'whale_64299' ,'whale_54850' ,'whale_38191' ,'whale_64937' ,'whale_14270' ,'whale_58474' ,'whale_21160' ,'whale_13863' ,'whale_51603' ,'whale_38543' ,'whale_29858' ,'whale_88547' ,'whale_07647' ,'whale_94546' ,'whale_24730' ,'whale_58362' ,'whale_21655' ,'whale_74935' ,'whale_79193' ,'whale_69619' ,'whale_43045' ,'whale_92686' ,'whale_84963' ,'whale_11625' ,'whale_66353' ,'whale_34663' ,'whale_45465' ,'whale_44699' ,'whale_64903' ,'whale_52759' ,'whale_55333' ,'whale_53889' ,'whale_78565' ,'whale_14094' ,'whale_57338' ,'whale_51195' ,'whale_41805' ,'whale_86527' ,'whale_45728' ,'whale_32702' ,'whale_45367' ,'whale_55861' ,'whale_98151' ,'whale_14892' ,'whale_68774' ,'whale_73666' ,'whale_21873' ,'whale_08923' ,'whale_22059' ,'whale_67614' ,'whale_76398' ,'whale_33195' ,'whale_13789' ,'whale_11099' ,'whale_25715' ,'whale_88756' ,'whale_10021' ,'whale_17601' ,'whale_82602' ,'whale_58972' ,'whale_86585' ,'whale_04540' ,'whale_03728' ,'whale_67801' ,'whale_48497' ,'whale_86158' ,'whale_04435' ,'whale_39674' ,'whale_88746' ,'whale_98746' ,'whale_75455' ,'whale_04397' ,'whale_33961' ,'whale_74232' ,'whale_66205' ,'whale_58747' ,'whale_79199' ,'whale_27834' ,'whale_63816' ,'whale_15078' ,'whale_04373' ,'whale_92515' ,'whale_81768' ,'whale_08700' ,'whale_52998' ,'whale_38906' ,'whale_07808' ,'whale_23855' ,'whale_89456' ,'whale_32198' ,'whale_18845' ,'whale_42191' ,'whale_15615' ,'whale_90446' ,'whale_89541' ,'whale_82387' ,'whale_81960' ,'whale_18158' ,'whale_06967' ,'whale_72235' ,'whale_90244' ,'whale_67611' ,'whale_12609' ,'whale_98507' ,'whale_37301' ,'whale_49877' ,'whale_69084' ,'whale_41776' ,'whale_71062' ,'whale_49832' ,'whale_64526' ,'whale_30074' ,'whale_61728' ,'whale_78372' ,'whale_47858' ,'whale_65263' ,'whale_38302' ,'whale_27221' ,'whale_56281' ,'whale_24458' ,'whale_43961' ,'whale_86081' ,'whale_21213' ,'whale_80124' ,'whale_00442' ,'whale_78395' ,'whale_35004' ,'whale_99326' ,'whale_99243' ,'whale_92153' ,'whale_75215' ,'whale_40190' ,'whale_05784' ,'whale_41881' ,'whale_74062' ,'whale_62655' ,'whale_46747' ,'whale_34798' ,'whale_90377' ,'whale_43326' ,'whale_20248' ,'whale_04480' ,'whale_35430' ,'whale_25659' ,'whale_10583' ,'whale_40885' ,'whale_28384' ,'whale_22212' ,'whale_06997' ,'whale_49277' ,'whale_22101' ,'whale_60451' ,'whale_41921' ,'whale_96240' ,'whale_79948' ,'whale_17528' ,'whale_88085' ,'whale_02608' ,'whale_12085' ,'whale_32021' ,'whale_88226' ,'whale_44071' ,'whale_48633' ,'whale_19027' ,'whale_90966' ,'whale_58309' ,'whale_78628' ,'whale_36648' ,'whale_37014' ,'whale_29294' ,'whale_08324' ,'whale_67407' ,'whale_88478' ,'whale_40483' ,'whale_71554' ,'whale_68338' ,'whale_28216' ,'whale_84264' ,'whale_52100' ,'whale_49210' ,'whale_75413' ,'whale_23467' ,'whale_98633' ,'whale_43374' ,'whale_83157' ,'whale_47062' ,'whale_51332' ,'whale_31739' ,'whale_61924' ,'whale_66539' ,'whale_12661' ,'whale_38437' ,'whale_27860' ,'whale_97440' ,'whale_14626' ,'whale_90141' ,'whale_16576' ,'whale_77984' ,'whale_24679' ,'whale_66421' ,'whale_07483' ,'whale_22118' ,'whale_64274' ,'whale_98618' ,'whale_03935' ,'whale_87420' ,'whale_69943' ,'whale_47768' ,'whale_03990' ,'whale_09422' ,'whale_07331' ,'whale_88432' ,'whale_66935' ,'whale_45294' ,'whale_74828' ,'whale_13701' ,'whale_54497' ,'whale_65737' ,'whale_16406' ,'whale_75767' ,'whale_26212' ,'whale_48024' ,'whale_73167' ,'whale_34813' ,'whale_34513' ,'whale_81915' ,'whale_44747' ,'whale_16762' ,'whale_22848' ,'whale_17327' ,'whale_89271' ,'whale_08729' ,'whale_05140' ,'whale_51114' )

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

def get_max(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    max_inds = 0
    max_score = 0.0
    if len(inds) == 0:
   #     print('Warning: no target detected!')
        return
    elif len(inds) > 1:
  #      print('Warning: ' + str(len(inds)) + ' targets detected! Choose the highest one')
        for i in inds:
            if(dets[i, -1] > max_score):
                max_inds = i
                max_score = dets[i, -1]
    bbox = dets[max_inds, :4]
    score = dets[max_inds, -1]
    return [max_inds,score]

def vis_detections(im, class_name, dets, image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    max_inds = 0
    max_score = 0.0
    if len(inds) == 0:
   #     print('Warning: no target detected!')
        return
    elif len(inds) > 1:
  #      print('Warning: ' + str(len(inds)) + ' targets detected! Choose the highest one')
        for i in inds:
            if(dets[i, -1] > max_score):
                max_inds = i
                max_score = dets[i, -1]

#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
    # for i in inds:
    # bbox = dets[i, :4]
    # score = dets[i, -1]
    #print max_inds
    bbox = dets[max_inds, :4]
    score = dets[max_inds, -1]

#    ax.add_patch(
#        plt.Rectangle((bbox[0], bbox[1]),
#                        bbox[2] - bbox[0],
#                        bbox[3] - bbox[1], fill=False,
#                        edgecolor='red', linewidth=3.5)
#        )
#    ax.text(bbox[0], bbox[1] - 2,
#            '{:s} {:.3f}'.format(class_name, score),
#            bbox=dict(facecolor='blue', alpha=0.5),
#            fontsize=14, color='white')

    # end for
    #print image_name, class_name
    #print score
   # file.writelines([image_name,'\t',class_name,'\t',str(score),'\n'])
 #   ax.set_title(('{} detections with '
 #                 'p({} | box) >= {:.1f}').format(class_name, class_name,
 #                                                 thresh),fontsize=14)
 #   plt.axis('off')
 #   plt.tight_layout()
 #   plt.draw()
	### SAVE IMAGES ? ###
    save_img_dir = os.path.join(cfg.ROOT_DIR, 'result', 'test_img')
  #  if not os.path.exists(save_img_dir):
  #      os.makedirs(save_img_dir)
   # plt.savefig(os.path.join(save_img_dir, image_name + '_' + class_name))

    boxes = {'boxes': ((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1])}
    
    save_mat_dir = os.path.join(cfg.ROOT_DIR, 'result', 'test_box')
   # if not os.path.exists(save_mat_dir):
  #      os.makedirs(save_mat_dir)
   # sio.savemat(os.path.join(save_mat_dir, image_name + '.mat'), {'boxes': boxes})


def demo(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    # box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',image_name + '_boxes.mat')
    test_mats_path = '/home/tanshen/fast-rcnn/data/kaggle/test_bbox'
    box_file = os.path.join(test_mats_path ,image_name + '_boxes.mat')
    obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    test_images_path = '/home/tanshen/fast-rcnn/data/kaggle/ImagesTest'
    # im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im_file = os.path.join(test_images_path, image_name + '.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
   # print ('Detection took {:.3f}s for '
   #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0
    NMS_THRESH = 0.3
    max_inds = 0
    max_score = 0.0
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
       # print 'All {} detections with p({} | box) >= {:.1f} in {}'.format(cls, cls,
       #                                                             CONF_THRESH, image_name)
        #if get_max!=[]: 

        [ind,tmp]=get_max(im, cls, dets, thresh=CONF_THRESH)
        #print image_name,cls,tmp

        #vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)
        #print dets[:,-1]
    #print image_name,max_score
        file.writelines([image_name,'\t',cls,'\t',str(tmp),'\n'])
        if(max_score<tmp):
            max_score=tmp
            cls_max=cls
    print image_name,cls_max,max_score

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [caffenet]',
                        choices=NETS.keys(), default='vgg_cnn_m_1024')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    file = open('result_ImageTest_All_vgg_cnn_m_1024.txt', 'w')

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # print 'Demo for data/demo/w_11107.jpg'

    test_images_path = '/home/tanshen/fast-rcnn/data/kaggle/ImagesTest'
    # length = len([name for name in os.listdir(test_images_path) if os.path.isfile(os.path.join(test_images_path, name))])
    cnt=1
    for name in os.listdir(test_images_path):
        if os.path.isfile(os.path.join(test_images_path, name)):
            demo(net, name.replace('.jpg',''), ('whale_48813','whale_09913','whale_45062','whale_74162','whale_99558' ,'whale_59255' ,'whale_87291' ,'whale_33152' ,'whale_88147' ,'whale_77693' ,'whale_74625' ,'whale_06069' ,'whale_62939' ,'whale_67036' ,'whale_78785' ,'whale_75682' ,'whale_08017' ,'whale_23574' ,'whale_46169' ,'whale_11242' ,'whale_03227' ,'whale_24815' ,'whale_36231' ,'whale_89615' ,'whale_64989' ,'whale_52342' ,'whale_63948' ,'whale_80947' ,'whale_80247' ,'whale_11076' ,'whale_70138' ,'whale_07863' ,'whale_36300' ,'whale_64634' ,'whale_22448' ,'whale_68116' ,'whale_30331' ,'whale_64006' ,'whale_54796' ,'whale_48415' ,'whale_09181' ,'whale_86041' ,'whale_81818' ,'whale_99573' ,'whale_47734' ,'whale_58010' ,'whale_23525' ,'whale_48490' ,'whale_02411' ,'whale_08439' ,'whale_08181' ,'whale_78280' ,'whale_23847' ,'whale_29569' ,'whale_67685' ,'whale_49530' ,'whale_48386' ,'whale_49491' ,'whale_39915' ,'whale_38008' ,'whale_50021' ,'whale_81875' ,'whale_10977' ,'whale_82548' ,'whale_36154' ,'whale_85670' ,'whale_23367' ,'whale_98645' ,'whale_87155' ,'whale_59173' ,'whale_74683' ,'whale_76782' ,'whale_73592' ,'whale_00195' ,'whale_43971' ,'whale_79439' ,'whale_53079' ,'whale_52505' ,'whale_97882' ,'whale_95370' ,'whale_94176' ,'whale_63541' ,'whale_70904' ,'whale_28892' ,'whale_16738' ,'whale_05487' ,'whale_83892' ,'whale_57251' ,'whale_72820' ,'whale_98996' ,'whale_12820' ,'whale_66852' ,'whale_27085' ,'whale_90820' ,'whale_23821' ,'whale_06339' ,'whale_83791' ,'whale_37154' ,'whale_87622' ,'whale_40169' ,'whale_15079' ,'whale_52749' ,'whale_11708' ,'whale_82064' ,'whale_60729' ,'whale_17604' ,'whale_28263' ,'whale_19906' ,'whale_34656' ,'whale_61461' ,'whale_96385' ,'whale_73684' ,'whale_23118' ,'whale_87604' ,'whale_37654' ,'whale_39689' ,'whale_26288' ,'whale_37658' ,'whale_55550' ,'whale_69459' ,'whale_90929' ,'whale_35594' ,'whale_41125' ,'whale_79823' ,'whale_46974' ,'whale_35844' ,'whale_03623' ,'whale_18989' ,'whale_47700' ,'whale_37269' ,'whale_85464' ,'whale_53580' ,'whale_38681' ,'whale_54574' ,'whale_44127' ,'whale_73080' ,'whale_61260' ,'whale_48966' ,'whale_64714' ,'whale_55079' ,'whale_98939' ,'whale_26397' ,'whale_91826' ,'whale_64496' ,'whale_17785' ,'whale_73136' ,'whale_49237' ,'whale_68789' ,'whale_20448' ,'whale_31594' ,'whale_27820' ,'whale_19041' ,'whale_06334' ,'whale_32087' ,'whale_82089' ,'whale_74439' ,'whale_60921' ,'whale_65057' ,'whale_75932' ,'whale_97542' ,'whale_02839' ,'whale_36851' ,'whale_25878' ,'whale_15434' ,'whale_89211' ,'whale_09454' ,'whale_29172' ,'whale_08637' ,'whale_35426' ,'whale_86377' ,'whale_51538' ,'whale_66711' ,'whale_80405' ,'whale_05349' ,'whale_97688' ,'whale_09651' ,'whale_26686' ,'whale_54920' ,'whale_90271' ,'whale_36437' ,'whale_65586' ,'whale_33723' ,'whale_79648' ,'whale_09062' ,'whale_59627' ,'whale_97924' ,'whale_38817' ,'whale_05661' ,'whale_11555' ,'whale_13288' ,'whale_49135' ,'whale_92465' ,'whale_90957' ,'whale_12074' ,'whale_34488' ,'whale_79166' ,'whale_33140' ,'whale_86206' ,'whale_82554' ,'whale_26654' ,'whale_22297' ,'whale_95831' ,'whale_15519' ,'whale_38288' ,'whale_72327' ,'whale_90911' ,'whale_10005' ,'whale_38894' ,'whale_03103' ,'whale_81136' ,'whale_84178' ,'whale_95091' ,'whale_39293' ,'whale_64299' ,'whale_54850' ,'whale_38191' ,'whale_64937' ,'whale_14270' ,'whale_58474' ,'whale_21160' ,'whale_13863' ,'whale_51603' ,'whale_38543' ,'whale_29858' ,'whale_88547' ,'whale_07647' ,'whale_94546' ,'whale_24730' ,'whale_58362' ,'whale_21655' ,'whale_74935' ,'whale_79193' ,'whale_69619' ,'whale_43045' ,'whale_92686' ,'whale_84963' ,'whale_11625' ,'whale_66353' ,'whale_34663' ,'whale_45465' ,'whale_44699' ,'whale_64903' ,'whale_52759' ,'whale_55333' ,'whale_53889' ,'whale_78565' ,'whale_14094' ,'whale_57338' ,'whale_51195' ,'whale_41805' ,'whale_86527' ,'whale_45728' ,'whale_32702' ,'whale_45367' ,'whale_55861' ,'whale_98151' ,'whale_14892' ,'whale_68774' ,'whale_73666' ,'whale_21873' ,'whale_08923' ,'whale_22059' ,'whale_67614' ,'whale_76398' ,'whale_33195' ,'whale_13789' ,'whale_11099' ,'whale_25715' ,'whale_88756' ,'whale_10021' ,'whale_17601' ,'whale_82602' ,'whale_58972' ,'whale_86585' ,'whale_04540' ,'whale_03728' ,'whale_67801' ,'whale_48497' ,'whale_86158' ,'whale_04435' ,'whale_39674' ,'whale_88746' ,'whale_98746' ,'whale_75455' ,'whale_04397' ,'whale_33961' ,'whale_74232' ,'whale_66205' ,'whale_58747' ,'whale_79199' ,'whale_27834' ,'whale_63816' ,'whale_15078' ,'whale_04373' ,'whale_92515' ,'whale_81768' ,'whale_08700' ,'whale_52998' ,'whale_38906' ,'whale_07808' ,'whale_23855' ,'whale_89456' ,'whale_32198' ,'whale_18845' ,'whale_42191' ,'whale_15615' ,'whale_90446' ,'whale_89541' ,'whale_82387' ,'whale_81960' ,'whale_18158' ,'whale_06967' ,'whale_72235' ,'whale_90244' ,'whale_67611' ,'whale_12609' ,'whale_98507' ,'whale_37301' ,'whale_49877' ,'whale_69084' ,'whale_41776' ,'whale_71062' ,'whale_49832' ,'whale_64526' ,'whale_30074' ,'whale_61728' ,'whale_78372' ,'whale_47858' ,'whale_65263' ,'whale_38302' ,'whale_27221' ,'whale_56281' ,'whale_24458' ,'whale_43961' ,'whale_86081' ,'whale_21213' ,'whale_80124' ,'whale_00442' ,'whale_78395' ,'whale_35004' ,'whale_99326' ,'whale_99243' ,'whale_92153' ,'whale_75215' ,'whale_40190' ,'whale_05784' ,'whale_41881' ,'whale_74062' ,'whale_62655' ,'whale_46747' ,'whale_34798' ,'whale_90377' ,'whale_43326' ,'whale_20248' ,'whale_04480' ,'whale_35430' ,'whale_25659' ,'whale_10583' ,'whale_40885' ,'whale_28384' ,'whale_22212' ,'whale_06997' ,'whale_49277' ,'whale_22101' ,'whale_60451' ,'whale_41921' ,'whale_96240' ,'whale_79948' ,'whale_17528' ,'whale_88085' ,'whale_02608' ,'whale_12085' ,'whale_32021' ,'whale_88226' ,'whale_44071' ,'whale_48633' ,'whale_19027' ,'whale_90966' ,'whale_58309' ,'whale_78628' ,'whale_36648' ,'whale_37014' ,'whale_29294' ,'whale_08324' ,'whale_67407' ,'whale_88478' ,'whale_40483' ,'whale_71554' ,'whale_68338' ,'whale_28216' ,'whale_84264' ,'whale_52100' ,'whale_49210' ,'whale_75413' ,'whale_23467' ,'whale_98633' ,'whale_43374' ,'whale_83157' ,'whale_47062' ,'whale_51332' ,'whale_31739' ,'whale_61924' ,'whale_66539' ,'whale_12661' ,'whale_38437' ,'whale_27860' ,'whale_97440' ,'whale_14626' ,'whale_90141' ,'whale_16576' ,'whale_77984' ,'whale_24679' ,'whale_66421' ,'whale_07483' ,'whale_22118' ,'whale_64274' ,'whale_98618' ,'whale_03935' ,'whale_87420' ,'whale_69943' ,'whale_47768' ,'whale_03990' ,'whale_09422' ,'whale_07331' ,'whale_88432' ,'whale_66935' ,'whale_45294' ,'whale_74828' ,'whale_13701' ,'whale_54497' ,'whale_65737' ,'whale_16406' ,'whale_75767' ,'whale_26212' ,'whale_48024' ,'whale_73167' ,'whale_34813' ,'whale_34513' ,'whale_81915' ,'whale_44747' ,'whale_16762' ,'whale_22848' ,'whale_17327' ,'whale_89271' ,'whale_08729' ,'whale_05140' ,'whale_51114', ))
            if cnt%100==0:
                print cnt/6925*100,"%"
            cnt=cnt+1
            plt.close()
            # plt.show()
