#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import json
import time
import os
import signal
import subprocess
import cv2
from workerlog import *
from mysqlop import *
from pynvml import *
# sudo pip install nvidia-ml-py

from DetectionTrain.yolo2.preparedata import *
from DetectionTrain.yolo2.voc_label import *
from DetectionTrain.yolo2.modifyclassnum import *
from plotacc import *

from ClassifyTrain.prepareclsdata import *

DONE = 0
PROCESS = 1
GENDONE = 2
ERROR = 3

paramfile = 'trainparams.json'
configfile = './config.json'
jsondata = json.load(file(configfile))
gpuid = jsondata['gpuid']
#usrname = jsondata['usrname']
workername = jsondata['workername']+str(gpuid)
rootpath = jsondata['rootpath']
serverpath = jsondata['serverpath']
logfile = 'test.log'
# trainratio = 0.8
# testratio = 1-trainratio

structuresize = {"mobilenet":224,
                            "shufflenet":224,
                            "googlenet":300,
                            "densenet121":224,
                            "densenet161":224,
                            "densenet169":224,
                            "densenet201":224,
                            "inception_bn":224,
                            "inception_resnet_v2":224,
                            "inception_v3":299,
                            "resnet18":224, 
                            "resnet34":224, 
                            "resnet50":224,
                            "resnet101":224,
                            "resnext18":224,
                            "resnext34":224,
                            "resnext50":224,
                            "resnext101":224,
                            "squeezenet":224,
                            "vgg11":224,
                            "vgg16":224,
                            "vgg19":224,
                            "alexnet":224,
                            "lenet":28,
                            "inception_bn_28":28,
                            "resnet20_28":28,
                            "resnet38_28":28,
                            "resnet56_28":28,
                            "resnext20_28":28,
                            "resnext38_28":28,
                            "resnext56_28":28}

epochs = 1000
# epochs = 5000
# batchsize = 64
# netstructure = 'yolo2'
# learningrate = 0.0001
# momentum = 0.9
# weightdecay = 0.0005

# epochs = 200
# batchsize = 128
# netstructure = 'mobilenet'
# learningrate = 0.05
# momentum = 0.9
# weightdecay = 0.0005

class Worker(object):
    def __init__(self):
        # Set up worker.
        self.status = DONE
        self.interval = 30
        self.pid = 0
        self.proc = None
        self.taskname = "None"
        self.usrname = "all"
        self.title = 'train'
        self.structure = 'none'
        self.errflag = False
        self.copyfile = PROCESS
        self.WorkerReg()
        while True:
            # 定时发送请求，向mysql服务器请求任务
            if self.status==DONE:
                info = self.getTask()
                info_data = json.loads(info)          
            time.sleep(self.interval)
            self.WorkerStatus(info_data)

    def checkIfTakeTask(self, workername):
        db = Database()
        workerlist = db.qureyWorkerStatus()
        maxmem = 0
        maxname = ''
        ownerworkerlist=[]
        allworkerlist=[]
        for worker in workerlist:
            # name, status, Memusage, ownername
            if worker[3]==self.usrname:
                ownerworkerlist.append(worker)
            if worker[3]=="all":
                allworkerlist.append(worker)

        if self.usrname != "all" and len(ownerworkerlist)>0:
            for worker in ownerworkerlist:
                # name, status, Memusage, ownername
                if worker[1]==0 and worker[2]>maxmem:
                    maxname = worker[0]
                    maxmem = worker[2]
            print maxname, maxmem, self.usrname
            if maxname == workername:
                return True
            else:
                return False

        if self.usrname == "all" and (len(allworkerlist)>0):
            for worker in allworkerlist:
                # name, status, Memusage, ownername
                if worker[1]==0 and worker[2]>maxmem:
                    maxname = worker[0]
                    maxmem = worker[2]
            print maxname, maxmem, self.usrname
            if maxname == workername:
                return True
            else:
                return False

        return False

    def getTask(self):
        db = Database()
        res = db.qureyWorkerOwner(workername)
        if len(res)>0:
            ownername = res[0][0]
        else:
            ownername = 'all'
        self.usrname  = ownername
        tasklist = db.qureyTaskname(ownername)
        flag = self.checkIfTakeTask(workername)
        if flag:
            for task in tasklist:  # name, createtime, process, status, type
                string = '{"status": "todo", "name": "%s", "usrname": "%s", "type": %d}'%(task[0], task[5], task[4])
                print string
                info_data = json.loads(string)
                self.taskname = info_data['name']
                db.startTaskTrain(info_data['usrname'], info_data['name'])
                if(os.path.exists(logfile)):
                    os.remove(logfile)
                #print taskstatus
                print 'start train', info_data['name'], info_data['usrname']
                self.status = PROCESS
                self.usrname = info_data['usrname']
                self.startWorker(info_data)
                return string
                # if task[3]==2: # and task[4] == 0:
                #      string = '{"status": "doing", "name": "%s", "usrname": "%s"}'%(task[0], usrname)
                #      return string
        string = '{"status": "None"}'
        return string

    def startWorker(self, info):
        # 开启两个线程，一个处理任务，一个发送状态
        threads = []
        t1 = threading.Thread(target=self.process,args=(info,))
        threads.append(t1)
        #t2 = threading.Thread(target=self.WorkerStatus,args=(info,))
        #threads.append(t2)
        for t in threads:
            t.start()

    def process(self, info):
        usrname = info['usrname']
        taskname = info['name']
        tasktype = info['type']
        datapath = rootpath+usrname+'/'+taskname
        modelpath = rootpath+usrname+'/'+taskname + '/train'
        parampath = rootpath+usrname+'/'+taskname + '/' + paramfile
        global epochs
        
        if tasktype==0:
            epochs = 5000
            batchsize = 64
            netstructure = 'yolo2'
            learningrate = 0.0001
            momentum = 0.9
            weightdecay = 0.0005
            optimizer = 'SGD'
            Retrain = 0
            pretrainmodel = ''
            if os.path.exists(parampath):
                paramdata = json.load(file(parampath))
                epochs = paramdata['epoch']
                batchsize = int(paramdata['batchsize'])
                netstructure = paramdata['structure']
                optimizer = paramdata['optimizer']
                learningrate = float(paramdata['learningrate'])
                momentum = float(paramdata['momentum'])
                weightdecay = float(paramdata['weightdecay'])
                trainvalratio = int(paramdata['trainvalratio'])
                Retrain = int(paramdata['Retrain'])
                if Retrain == 1:
                    pretrainmodel = paramdata['pretrainmodel']
                    if pretrainmodel == '':
                        Retrain = 0
            #self.title = "%s %s  \n%s batch: %d, lr: %f"%(usrname, taskname, netstructure, int(batchsize), float(learningrate))
            self.title = "%s %s  \n%s batch: %d lr: %f, opt: %s"%(usrname, taskname, netstructure, batchsize, float(learningrate), optimizer)
            self.structure = netstructure
            self.copyfile = PROCESS
            trainnum = preparedata(datapath, trainvalratio)
            self.copyfile = DONE
            voclabel()
            argument = ''
            if netstructure=='yolo2':
                modifyclassnum(modelpath+'/'+netstructure, epochs, batchsize, learningrate, momentum, weightdecay)
                if Retrain==0:
                    argument = " detector train ./DetectionTrain/yolo2/voc.data ./DetectionTrain/yolo2/yolo.cfg ./DetectionTrain/yolo2/model/darknet19_448.conv.23 -gpus %d > %s"%(gpuid, logfile)
                else:
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodel
                    argument = " detector train ./DetectionTrain/yolo2/voc.data ./DetectionTrain/yolo2/yolo.cfg %s -gpus %d > %s"%(pretrainmodelpath, gpuid,  logfile)
                print argument
                proc = subprocess.Popen("./DetectionTrain/yolo2/darknet" + argument, shell=True)
                # proc = subprocess.Popen("./DetectionTrain/yolo2/darknet" + argument, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True)
            if 'SSD' in netstructure:
                cmd = "python DetectionTrain/ssd/tools/prepare_dataset.py --dataset pascal --year 2007 --set trainval --target ./data/train.lst --root ./data/VOCdevkit"
                os.system(cmd)
                cmd = "python DetectionTrain/ssd/tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target ./data/val.lst --shuffle False --root ./data/VOCdevkit"
                os.system(cmd)
            if netstructure=='SSD_VGG16_300x300':
                if Retrain==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network vgg16_reduced --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/vgg16_reduced --epoch 0 --prefix %s --optimizer %s --data-shape 300 --lr %f --momentum %f --wd %f  --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network vgg16_reduced --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 300 --lr %f --momentum %f --wd %f  --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
            if netstructure=='SSD_VGG16_512x512':
                if Retrain ==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network vgg16_reduced --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/vgg16_reduced --epoch 0 --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network vgg16_reduced --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
            if netstructure=='SSD_Inception_v3_512x512':
                if Retrain ==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network inceptionv3 --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/inception-v3 --epoch 0 --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network inceptionv3 --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                # proc = subprocess.Popen("python" + argument, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True)
                proc = subprocess.Popen("python" + argument, shell=True)
            if netstructure=='SSD_Resnet50_512x512':
                if Retrain ==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network resnet50 --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/resnet-50 --epoch 0 --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network resnet50 --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
            if netstructure=='SSD_Resnet101_512x512':
                if Retrain ==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network resnet101 --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/resnet-101 --epoch 0 --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network resnet101 --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
            if netstructure=='SSD_Mobilenet_300x300':
                if Retrain ==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network mobilenet --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/mobilenet --epoch 0 --prefix %s --optimizer %s --data-shape 300 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network mobilenet --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 300 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
            if netstructure=='SSD_Mobilenet_512x512':
                if Retrain ==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network mobilenet --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/mobilenet --epoch 0 --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network mobilenet --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 512 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
            if netstructure=='SSD_Mobilenet_608x608':
                if Retrain ==0:
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network mobilenet --end-epoch %d --batch-size %d --pretrained DetectionTrain/ssd/model/mobilenet --epoch 0 --prefix %s --optimizer %s --data-shape 608 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, int(batchsize), modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                else:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = " DetectionTrain/ssd/train.py --gpus %d --network mobilenet --begin-epoch %d --end-epoch %d --batch-size %d --pretrained %s --epoch %d --prefix %s --optimizer %s --data-shape 608 --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, startepoch, startepoch+epochs, int(batchsize), pretrainmodelpath, startepoch, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
            # if netstructure=="Faster_rcnn_VGG":
            #     argument = " DetectionTrain/rcnn/train_end2end.py --gpus %d --end_epoch %d --network vgg --pretrained DetectionTrain/rcnn/model/vgg16 --pretrained_epoch 0 --prefix %s --optimizer %s --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
            #     print argument
            #     proc = subprocess.Popen("python" + argument, shell=True)
            # if netstructure=="Faster_rcnn_Resnet101":
            #     argument = " DetectionTrain/rcnn/train_end2end.py --gpus %d --end_epoch %d --network resnet --pretrained DetectionTrain/rcnn/model/resnet-101 --pretrained_epoch 0 --prefix %s --optimizer %s --lr %f --momentum %f --wd %f --num-example %d --log %s"%(gpuid, epochs, modelpath+'/'+netstructure+'/'+taskname, optimizer, learningrate, momentum, weightdecay, trainnum, logfile)
            #     print argument
            #     proc = subprocess.Popen("python" + argument, shell=True)
            # line = proc.stdout.readline()
            # returncode = proc.poll()
            # line = line.strip()
            # print line
        else:
            if tasktype==1:
                epochs = 200
                batchsize = 128
                netstructure = 'mobilenet'
                learningrate = 0.05
                momentum = 0.9
                weightdecay = 0.0005
                optimizer = 'SGD'
                pretrainmodel = ''
                trainvalratio = 5
                Retrain = 0
                if os.path.exists(parampath):
                    paramdata = json.load(file(parampath))
                    epochs = paramdata['epoch']
                    batchsize = int(paramdata['batchsize'])
                    netstructure = paramdata['structure']
                    optimizer = paramdata['optimizer']
                    learningrate = float(paramdata['learningrate'])
                    momentum = float(paramdata['momentum'])
                    weightdecay = float(paramdata['weightdecay'])
                    Retrain = int(paramdata['Retrain'])
                    trainvalratio = int(paramdata['trainvalratio'])
                    if Retrain == 1:
                        pretrainmodel = paramdata['pretrainmodel']
                        if pretrainmodel == '':
                            Retrain = 0
                print epochs, batchsize, netstructure, learningrate, momentum, weightdecay
                self.title = "%s %s \n%s batch: %d, lr: %f, opt: %s"%(usrname, taskname, netstructure, int(batchsize), float(learningrate),optimizer)
                self.structure = netstructure
                self.copyfile = PROCESS
                classnum, filenum = prepareclsdata(datapath, trainvalratio)
                trainratio = trainvalratio*1.0/(trainvalratio+1)
                testratio = 1.0/(trainvalratio+1)
                self.copyfile = DONE
                cmd = "python ClassifyTrain/im2rec.py --train-ratio %f --test-ratio %f --list=True --recursive=True data ./data/"%(trainratio, testratio)
                os.system(cmd)
                cmd = "mv data_train.lst data/train.lst"
                os.system(cmd)
                cmd = "mv data_test.lst data/test.lst"
                os.system(cmd)
                size = structuresize[netstructure]
                cmd = "rm mean.bin"
                os.system(cmd)
                cmd = "python ClassifyTrain/im2rec.py --num-thread 4 --resize %d data data"%(size)
                os.system(cmd)
                #modelpath.replace(rootpath, serverpath)
                argument = " ClassifyTrain/train.py --num_classes %d --structure %s --num_examples %d --gpus %d --epoch %d --batch_size %d --lr %f --momentum %f --wd %f --log_file %s --optimizer %s --model_save_prefix %s"%(classnum, netstructure, filenum, gpuid, epochs, batchsize, learningrate, momentum, weightdecay, logfile, optimizer, modelpath+'/'+netstructure+'/'+netstructure)
                if Retrain==1:
                    pretrainmodelprefix = pretrainmodel.split('.')[0].split('-')[0]
                    startepoch = int(pretrainmodel.split('.')[0].split('-')[-1])
                    pretrainmodelpath = modelpath+'/'+netstructure+'/'+pretrainmodelprefix
                    argument = argument + ' --model_load_prefix %s --model_load_epoch %d'%(pretrainmodelpath, startepoch)
                print argument
                proc = subprocess.Popen("python" + argument, shell=True)
        time.sleep(1)
        self.errflag = False
        self.pid = proc.pid+1
        print "pid:", self.pid
        #cmd = "./darknet detector train voc.data yolo.cfg model/darknet19_448.conv.23 -gpus %d > test.log"%(gpuid)
        #os.system(cmd)
        db = Database()
        db.updateTask(usrname, taskname, 0.0, self.pid)
        self.proc = proc
        return

    def WorkerReg(self):
        db = Database()
        gpuinfo, memuse, memunused, util = self.queryGPUInfo()
        db.addWorker(workername, gpuinfo, memunused)
        return

    def WorkerStatus(self, info):
        db = Database()
        gpuinfo, memuse, memunused, util = self.queryGPUInfo()

        status = info['status']
        
        usrname = self.usrname
        taskname = self.taskname
        st = 0
        if memuse > 0.5:
            st = 1
        db.insertWorker(workername, st, taskname, gpuinfo, memunused)
        #print info
        res = db.getTaskStatus(usrname, taskname)
        print res
        taskstatus = 0
        stopflag = 0
        if len(res)>0:
            taskstatus = res[0][0]
            stopflag = res[0][1]
        if stopflag == 1:
            self.status=DONE
            self.taskname = "None"
            db.clearTask(usrname, taskname)
            db.clearWorker(taskname)
            if self.pid!=0 and self.pid!= 'None':
                cmd = "kill -9 "+str(self.pid)
                os.system(cmd)
                self.pid = 0
        if taskstatus>0 and taskname != "None":
            tasktype = info['type']
            # print tasktype
            st = 1
            # plot acc
            trainpath = rootpath+usrname+'/'+ taskname + '/train/'+self.structure+'/'
            ISOTIMEFORMAT='%Y%m%d%X'
            t = time.strftime( ISOTIMEFORMAT, time.localtime() )
            for file in os.listdir(trainpath):
                if file.split('.')[-1]=='jpg' or file.split('.')[-1]=='weights':  
                    file_path = os.path.join(trainpath, file)  
                    os.remove(file_path)
            figfile = trainpath+'train_'+str(t)+'.jpg'
            cmd = "cp %s %s"%(logfile,trainpath)
            os.system(cmd)
            binfile = trainpath+'mean.bin'
            if not os.path.exists(binfile):
                cmd = "cp mean.bin %s"%(binfile)
                print cmd
                os.system(cmd)
            wordsfile = trainpath+'words.txt'
            if not os.path.exists(wordsfile):
                cmd = "cp words.txt %s"%(wordsfile)
                print cmd
                os.system(cmd)
            global epochs
            #print self.proc
           
            # if self.errflag == False:
                # stat = self.proc.poll()
                # print 'stat:', stat, self.proc.pid
                # if stat is not None:
                #     stdout, stderr = self.proc.communicate()
                #     self.errflag = True
                #     flog=open(logfile, 'a+')
                #     flog.writelines(stdout)
                #     flog.flush()
                #     flog.close()
            per = plotresult(tasktype, epochs, logfile, figfile, False, self.title)
            if self.proc is not None:
                stat = self.proc.poll()
            else:
                stat = None
            # print stat
            if self.copyfile == DONE:
                if stat is not None and per < 1.0:
                    print 'cuda malloc failed, please check!!!'
                    flog=open(logfile, 'a+')
                    flog.writelines('cuda malloc failed, please check!!!\n')
                    flog.flush()
                    flog.close()

                if util==0 and memuse > 0.95:
                    print 'cuda do not work, please check!'
                    flog=open(logfile, 'a+')
                    flog.writelines('cuda do not work, please check!\n')
                    flog.flush()
                    flog.close()
            else:
                print 'copying file'
                flog=open(logfile, 'a+')
                flog.writelines('copying file\n')
                flog.flush()
                flog.close()

            if per >= 100.0: # todo differnet epochs give different result
                per = 100.0
            db.updateTask(usrname, taskname, per, self.pid)
            if per >= 100.0:
                print 'finish train', usrname, taskname
                db.FinishTaskTrain(usrname, taskname)
                self.status=DONE
                self.taskname = "None"   
                self.pid = 0
                usrname = info['usrname']
                taskname = info['name']
        else:
            self.taskname = "None"
            self.status=DONE
            self.pid = 0
        print workername, st, taskname, gpuinfo
        print self.status, self.pid, self.taskname
        db.updateWorker(workername, st, taskname, gpuinfo, memunused)
        return ;

    def queryGPUInfo(self):
        nvmlInit()
        # num = nvmlDeviceGetCount()
        # print num
        # for i in range(num):
        #     handle = nvmlDeviceGetHandleByIndex(i)
        #     info = nvmlDeviceGetMemoryInfo(handle)
        #     temp = nvmlDeviceGetTemperature(handle, 0)
        #     perc = nvmlDeviceGetFanSpeed(handle)
        #     power = nvmlDeviceGetPowerUsage(handle)
        #     util = nvmlDeviceGetUtilizationRates(handle)
        #     print "Device:", nvmlDeviceGetName(handle),  str(temp)+"C"
        #     print "Total memory:", info.total/1000000
        #     print "Used memory:", info.used/1000000
        #     print str(perc)+"%"
        #     print str(power/1000)+"W"
        #     print util.gpu, util.memory
        info = ""
        handle = nvmlDeviceGetHandleByIndex(gpuid)
        name = nvmlDeviceGetName(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)
        temp = nvmlDeviceGetTemperature(handle, 0)
        #perc = nvmlDeviceGetFanSpeed(handle)
        #power = nvmlDeviceGetPowerUsage(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        memuse = mem.used*1.0/mem.total 
        memunused = mem.total/1000000 - mem.used/1000000
        info = name + ": Mem: " + str(mem.used/1000000)+ "/"+ str(mem.total/1000000)+ " Temp: " + str(temp)+"C Util: " + str(util.gpu) 
        return info, memuse, memunused, util.gpu

if __name__ == "__main__":
    wk = Worker()
    #db.qureyTask()
    #db.writeTask('task1', 0.0, 'fj')
    # res = wk.queryGPUInfo()
    # print res
