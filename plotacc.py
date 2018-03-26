
import numpy as np  
import re  
import os
import argparse  
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  

STEP = 100
subbatch  =4
TRAINLINES = 1

def plotresult(tasktype, epochs, logfile, figfile, show, title):  
    percent = 0.0
    if(os.path.exists(logfile)):
        if tasktype==0:
            log = open(logfile).read()  
            if 'yolo' in title:
                TR_RE = re.compile('\sIOU: ([\d\.]+)')  
                VA_RE = re.compile('\sRecall: ([\d\.]+)') 
                #EPO_RE = re.compile('\sRecall: ([\d\.]+)') 
                try:  
                    log_iou0 = [float(x) for x in TR_RE.findall(log)]  
                    log_rec0 = [float(x) for x in VA_RE.findall(log)]  


                    #print len(log_iou0)
                    #print len(log_rec0)
                    percent = len(log_iou0)*100.0/(epochs*subbatch)

                    log_iou=[]
                    log_rec=[]
                    idx0 = np.arange(0, len(log_iou0), STEP) /subbatch
                    for i in idx0:
                        log_iou.append(log_iou0[i])

                    idx1 = np.arange(0, len(log_rec0), STEP) /subbatch
                    for i in idx1:
                        log_rec.append(log_rec0[i])

                    plt.figure(figsize=(8, 6))  
                    plt.axis([0.0,epochs,0.0,1.0])
                    plt.xlabel("Epoch")  
                    plt.ylabel("Avg IOU and Recall")  
                    plt.title(title)
                    plt.plot(idx0, log_iou, '.', linestyle='-', color="r",  label="IOU")  
                    plt.plot(idx1, log_rec, '.', linestyle='-', color="b", label="Recall")  
                    plt.legend(loc='best')
                    plt.savefig(figfile)
                    plt.close('all')
                except:
                    print 'logfile error'
            if 'SSD' in title:
                CrossEntropy_RE = re.compile('\sTrain-CrossEntropy=([\d\.]+)')  
                SmoothL1_RE = re.compile('\sTrain-SmoothL1=([\d\.]+)') 
                mAP_RE = re.compile('\sValidation-mAP=([\d\.]+)')  
                try:
                    log_crossentropy = [float(x) for x in CrossEntropy_RE.findall(log)]  
                    log_smoothl1 = [float(x) for x in SmoothL1_RE.findall(log)]  
                    log_map = [float(x) for x in mAP_RE.findall(log)]  

                    percent = len(log_map)*100.0/(epochs)

                    idx0 = np.arange(0, len(log_crossentropy))
                    idx1 = np.arange(0, len(log_smoothl1))
                    idx2 = np.arange(0, len(log_map))

                    # print log_crossentropy
                    # print log_smoothl1
                    # print log_map

                    plt.figure(figsize=(8, 6))  
                    plt.axis([0.0,epochs,0.0,1.5])
                    plt.xlabel("Epoch")  
                    plt.ylabel("CrossEntropy SmoothL1 and Val_mAP")  
                    plt.title(title)
                    plt.plot(idx0, log_crossentropy, '.', linestyle='-', color="r",  label="CrossEntropy")  
                    plt.plot(idx1, log_smoothl1, '.', linestyle='-', color="b", label="SmoothL1")  
                    plt.plot(idx2, log_map, '.', linestyle='-', color="g", label="Validation mAP")  
                    plt.legend(loc='best')
                    plt.savefig(figfile)
                    plt.close('all')
                except:
                    print 'logfile error'
            if 'Faster_rcnn' in title:
                RPNAcc_RE = re.compile('\sTrain-RPNAcc=([\d\.]+)')  
                RPNLogLoss_RE = re.compile('\sTrain-RPNLogLoss=([\d\.]+)') 
                RCNNAcc_RE = re.compile('\sTrain-RCNNAcc=([\d\.]+)')  
                RCNNLogLoss_RE = re.compile('\sTrain-RCNNLogLoss=([\d\.]+)') 

                log_rpnacc = [float(x) for x in RPNAcc_RE.findall(log)]  
                log_rpnloss = [float(x) for x in RPNLogLoss_RE.findall(log)]  
                log_rcnnacc = [float(x) for x in RCNNAcc_RE.findall(log)]  
                log_rcnnloss = [float(x) for x in RCNNLogLoss_RE.findall(log)]  
                #print len(log_iou0)
                #print len(log_rec0)
                percent = len(log_rpnacc)*100.0/(epochs)

                idx0 = np.arange(0, len(log_rpnacc))
                idx1 = np.arange(0, len(log_rpnloss))
                idx2 = np.arange(0, len(log_rcnnacc))
                idx3 = np.arange(0, len(log_rcnnloss))

                plt.figure(figsize=(8, 6))  
                plt.axis([0.0,epochs,0.0,1.5])
                plt.xlabel("Epoch")  
                plt.ylabel("RPN and RCNN Acc Loss")  
                plt.title(title)
                plt.plot(idx0, log_rpnacc, '.', linestyle='-', color="r",  label="RPN Acc")  
                plt.plot(idx1, log_rpnloss, '.', linestyle='-', color="b", label="RPN LogLoss")  
                plt.plot(idx2, log_rcnnacc, '.', linestyle='-', color="g", label="RCNN ACC")  
                plt.plot(idx3, log_rcnnloss, '.', linestyle='-', color="y", label="RCNN LogLoss")  
                plt.legend(loc='best')
                plt.savefig(figfile)
                plt.close('all')
        if tasktype==1:
            TR_RE = re.compile('.*?sec\s(accuracy|Train-accuracy)=([\d\.]+)')  
            #TR_RE = re.compile('.*?sec\sTrain-accuracy=([\d\.]+)')  
            VA_RE = re.compile('.*?]\sValidation-accuracy=([\d\.]+)')  
            log = open(logfile).read()  
            
            trstr = TR_RE.findall(log)
            log_tr = []
            for x in trstr:
                if len(x) == 2:
                    log_tr.append(float(x[1]))
                else:
                    log_tr.append(float(x))
            #log_tr = [float(x) for x in TR_RE.findall(log)]  
            log_va = [float(x) for x in VA_RE.findall(log)]  

            percent = len(log_va)*100.0/(epochs)

            idx = np.arange(len(log_tr))  
            idx1 = np.arange(len(log_va)) * TRAINLINES
              
            plt.figure(figsize=(8, 6))
            plt.axis([0.0,epochs,0.0,1.0])  
            plt.xlabel("Epoch")  
            plt.ylabel("Accuracy") 
            plt.title(title)
            plt.plot(idx, log_tr, 'o', linestyle='-', color="r",  label="Train accuracy")  
            plt.plot(idx1, log_va, 'o', linestyle='-', color="b",label="Validation accuracy")  
            plt.legend(loc="best")  
            plt.savefig(figfile)
            # plt.xticks(np.arange(min(idx), max(idx)+1, 5))  
            # plt.yticks(np.arange(0, 1, 0.2))  
            # plt.ylim([0,1])  
            plt.close('all')
    return percent

if __name__ == "__main__":
    epochs = 50
    logfile = 'test.log'
    figfile = 'test.jpg'
    title = 'Faster_rcnn\ngsfasf'
    per = plotresult(0, epochs, logfile, figfile, False, title)
    print per