from utils.data import *
from utils.metric import *
from argparse import ArgumentParser
import torch
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import *
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time

os.environ['CUDA_VISIBLE_DEVICES']="0"

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default='dataset/IRSTD1k')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default='weight/IRSTD-1k_weight.tar')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=True)
        self.val_loader = Data.DataLoader(valset, 1, drop_last=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # device = torch.device('cuda')
        self.device = device

        model = MSHNet(3)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        #记录训练log
        self.timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.result_dir = 'result'
        if not osp.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.result_csv_path = osp.join(self.result_dir, f'train_log_{self.timestamp}.csv')

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = ''
                checkpoint = torch.load(check_folder+'/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']+1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                self.save_folder = 'weights/MSHNet-%s'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                if not osp.exists(self.save_folder):
                    os.mkdir(self.save_folder)

                # ✅ [新增] 初始化 loss 日志文件路径
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                self.loss_log_file = osp.join(self.save_folder, f"loss_{timestamp}.txt")
                self.all_epoch_losses = []  # 存放每轮的平均 loss
        if args.mode=='test':
          
            weight = torch.load(args.weight_path)
            self.model.load_state_dict(weight['state_dict'])
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1
        

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()

        # 性能记录开始时间
        start_time = time.time()
        total_batch_time = 0

        tag = False
        for i, (data, mask) in enumerate(tbar):
            batch_start_time = time.time()  # 记录 batch 开始时间

            data = data.to(self.device)
            labels = mask.to(self.device)

            if epoch>self.warm_epoch:
                tag = True

            masks, pred = self.model(data, tag)
            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            for j in range(len(masks)):
                if j>0:
                    labels = self.down(labels)
                loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)
                
            loss = loss / (len(masks)+1)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))

            batch_end_time = time.time()  # 记录 batch 结束时间
            total_batch_time += batch_end_time - batch_start_time  # 累加 batch 时间
            # ✅ [新增] 记录当前 epoch 的 loss 值
            self.all_epoch_losses.append(losses.avg)

            with open(self.loss_log_file, "a") as f:
                f.write(f"{epoch + 1},{losses.avg:.6f}\n")

        end_time = time.time()
        epoch_time = end_time - start_time

        if len(self.train_loader) > 0:
            avg_batch_time = total_batch_time / len(self.train_loader)
            fps = self.args.batch_size / avg_batch_time
        else:
            avg_batch_time = 0
            fps = 0

        # 写入性能日志（ txt ）
        perf_log_path = osp.join(self.result_dir, f'train_perf_log_{self.timestamp}.txt')
        with open(perf_log_path, 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Epoch time: {epoch_time:.2f} s\n")
            f.write(f"  Avg batch time: {avg_batch_time * 1000:.2f} ms\n")
            f.write(f"  Train FPS: {fps:.2f} samples/sec\n\n")
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device)
                mask = mask.to(self.device)

                if epoch>self.warm_epoch:
                    tag = True

                loss = 0
                _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            
            if self.mode == 'train':

                # 训练指标写入以时间戳命名的 CSV 文件
                if not osp.exists(self.result_csv_path):
                    with open(self.result_csv_path, 'w') as f:
                        f.write("epoch,IoU,PD,FA\n")
                with open(self.result_csv_path, 'a') as f:
                    f.write(f"{epoch},{mean_IoU:.4f},{PD[0]:.4f},{FA[0] * 1e6:.4f}\n")

                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, PD[0], FA[0] * 1000000))
                        
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
                torch.save(all_states, self.save_folder+'/checkpoint.pkl')
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


         
if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)

        # ✅ [新增] 训练结束提示日志路径
        print(f"\n✅ Loss 日志已保存到：{trainer.loss_log_file}")
    else:
        trainer.test(1)
 