import numpy as np
import os
import torch

# import numpy as np
# import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import torch
from utils.config import args
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
import nets as models
from utils.bar_show import progress_bar
from src.noisydataset import cross_modal_dataset
import src.utils as utils
import scipy
import scipy.spatial
from src.bmm import BetaMixture1D
from src.loss import NGCEandMAE
from src.loss import NCEandRCE


best_acc = 0  # best test accuracy
start_epoch = 0

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.ckpt_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

def load_dict(model, path):
    chp = torch.load(path)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in chp['model_state_dict']:
            state_dict[key] = chp['model_state_dict'][key]
    model.load_state_dict(state_dict)

def main():
    print('===> Preparing data ..')
    train_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    valid_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'valid')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    test_dataset = cross_modal_dataset(args.data_name, args.noisy_ratio, 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    print('===> Building Models..')
    multi_models = []
    n_view = len(train_dataset.train_data)
    for v in range(n_view):
        if v == args.views.index('Img'): # Images
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())
        elif v == args.views.index('Txt'): # Text
            multi_models.append(models.__dict__['TextNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())
        else: # Default to use ImageNet
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())

    C = torch.Tensor(args.output_dim, args.output_dim)
    C = torch.nn.init.orthogonal(C, gain=1)[:, 0: train_dataset.class_num].cuda()
    C.requires_grad = True

    embedding = torch.eye(train_dataset.class_num).cuda()
    embedding.requires_grad = False

    parameters = [C]
    for v in range(n_view):
        parameters += list(multi_models[v].parameters())
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=args.lr, betas=[0.5, 0.999], weight_decay=args.wd)
    if args.ls == 'cos':
        lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)

    if args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss == 'MCE':
        criterion = utils.MeanClusteringError(train_dataset.class_num, None, tau=args.tau).cuda()
        criterion_no_mean = utils.MeanClusteringError(train_dataset.class_num, 1, tau=args.tau).cuda()
    elif args.loss == 'APL':
        criterion = NCEandRCE(1, 1, train_dataset.class_num)
        criterion_no_mean = NCEandRCE(1, 1, train_dataset.class_num, 1)
    elif args.loss == 'NGCEandMAE':
        criterion = NGCEandMAE(1, 1, train_dataset.class_num,0.7)
        criterion_no_mean = NGCEandMAE(1, 1, train_dataset.class_num,0.7, 1)
    else:
        raise Exception('No such loss function.')

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        for v in range(n_view):
            multi_models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    def set_train():
        for v in range(n_view):
            multi_models[v].train()

    def set_eval():
        for v in range(n_view):
            multi_models[v].eval()

    def cross_modal_contrastive_ctriterion(fea, tau=1.):
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())

        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss1 = -(diag1 / sim.sum(1)).log().mean()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss2 = -(diag2 / sim.sum(1)).log().mean()
        return loss1 + loss2

    def contrastive(fea, tar, tau=1.):
        loss = []
        for v in range(n_view):
            sim = fea[v].mm(fea[v].t())
            sim = (sim / tau).exp()
            dif = tar[v] - tar[v].reshape(-1, 1)
            condition1 = dif == 0
            condition2 = dif != 0
            masked_sim1 = torch.masked_fill(sim, condition1, value=0)  # not seem class
            masked_sim2 = torch.masked_fill(sim, condition2, value=0)  # seem class
            top_value, top_i = torch.topk(masked_sim2, 1)
            select_sim = (top_value.sum()).reshape(1, -1).squeeze()
            loss.append(-(select_sim / masked_sim1.sum(1)).log().mean())
        return loss[0] + loss[1]
    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train()
        select_idx = [torch.tensor([], dtype=np.int) for i in range(n_view)]
        train_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view

        for batch_idx, (batches, targets, index) in enumerate(train_loader):
            batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]
            index = index
            batch_size = batches[0].shape[0]

            norm = C.norm(dim=0, keepdim=True)
            C.data = (C / norm).detach()

            for v in range(n_view):
                multi_models[v].zero_grad()
            optimizer.zero_grad()

            outputs = [multi_models[v](batches[v]) for v in range(n_view)]
            preds = [outputs[v].mm(C) for v in range(n_view)]

            mceloss = [criterion_no_mean(preds[v], targets[v]) for v in range(n_view)]
            losses = [torch.mean(mceloss[v]) for v in range(n_view)]
            mceloss = torch.stack(mceloss).reshape(1, -1).squeeze()
            loss = sum(losses)

            loss_nor = (mceloss - mceloss.min()) / (mceloss.max() - mceloss.min())
            # loss_nor = (ce_loss - ce_loss.min()) / (ce_loss.max() - ce_loss.min())
            bmm_A = BetaMixture1D(max_iters=10)
            loss_i = loss_nor.reshape(-1, 1)[0:batch_size].cpu().detach().numpy()
            bmm_A.fit(loss_i)
            prob_A = bmm_A.posterior(loss_i, 0).T.squeeze()

            bmm_B = BetaMixture1D(max_iters=10)
            loss_t = loss_nor.reshape(-1, 1)[batch_size:].cpu().detach().numpy()
            bmm_B.fit(loss_t)
            prob_B = bmm_A.posterior(loss_t, 0).T.squeeze()

            # mix
            # select_num = len(prob_A) // 5
            # select_num = min(int(len(prob_A)*(args.noisy_ratio)*0.5), int(len(prob_A)*(1-args.noisy_ratio))) #筛选干净的数量
            select_num = int(len(prob_A)*(1-args.noisy_ratio)) #筛选干净的数量
            threshld_A = np.sort(prob_A)[::-1][select_num]
            threshld_B = np.sort(prob_B)[::-1][select_num]
            pred_A = (prob_A > threshld_A).squeeze()
            pred_B = (prob_B > threshld_B).squeeze()
            selected = [pred_A, pred_B]

            for v in range(n_view):
                ss = index[selected[v]]
                select_idx[v] = torch.cat([select_idx[v], ss], dim=0)

            inputs_x = [torch.tensor([], dtype=torch.float32) for i in range(n_view)]
            targets_x = [torch.tensor([], dtype=torch.float32) for i in range(n_view)]
            inputs_u = [torch.tensor([], dtype=torch.float32) for i in range(n_view)]
            targets_u = [torch.tensor([], dtype=torch.float32) for i in range(n_view)]
            all_inputs = [torch.tensor([], dtype=torch.float32) for i in range(n_view)]
            all_targets = [torch.tensor([], dtype=torch.float32) for v in range(n_view)]

            for v in range(n_view):
                inputs_x[v] = batches[v][selected[v]]
                targets_x[v] = torch.nn.functional.one_hot(targets[v][selected[v]], train_dataset.class_num).float()

                inputs_u[v] = batches[v][~selected[v]]
                targets_u[v] = torch.nn.functional.one_hot(targets[v][~selected[v]], train_dataset.class_num).float()

                u_size = inputs_u[v].size()[0]
                x_size = inputs_x[v].size()[0]
                if x_size != 0:
                    index = np.random.permutation(x_size)
                    lam = 0.1
                    for i in range(int(u_size)):
                        j = i % x_size
                        inputs_u[v][i, :] = lam * inputs_u[v][i, :] + (1 - lam) * inputs_x[v][index[j], :]
                        targets_u[v][i] = lam * targets_u[v][i] + (1 - lam) * targets_x[v][index[j]]

                # size = inputs_x[v].size()[0]
                # index = np.random.permutation(size)
                # lam = 0.2
                # for i in range(size*2):
                #     j = i % size
                #     inputs_u[v][i, :] = lam * inputs_u[v][i, :] + (1 - lam) * inputs_x[v][index[j], :]
                #     targets_u[v][i] = lam * targets_u[v][i] + (1 - lam) * targets_x[v][index[j]]

                # size = inputs_x[v].size()[0]
                # index = np.random.permutation(size)
                # lam = 0.05
                # for i in range(size):
                #     inputs_u[v][i, :] = lam * inputs_u[v][i, :] + (1 - lam) * inputs_x[v][index[i], :]
                #     targets_u[v][i] = lam * targets_u[v][i] + (1 - lam) * targets_x[v][index[i]]

                all_inputs[v] = torch.cat([inputs_x[v], inputs_u[v]], dim=0).cuda()
                all_targets[v] = torch.cat([targets_x[v], targets_u[v]], dim=0).cuda()



            outputs1 = [multi_models[v](all_inputs[v]) for v in range(n_view)]
            preds1 = [outputs1[v].mm(C) for v in range(n_view)]
            losses1 = [criterion(preds1[v], all_targets[v]) for v in range(n_view)]
            loss1 = sum(losses1)
            contrastiveLoss = cross_modal_contrastive_ctriterion(outputs, tau=args.tau)
            # contrastiveLoss = 0.2 * contrastive(outputs, targets, tau=args.tau) + cross_modal_contrastive_ctriterion(outputs, tau=args.tau)

            if epoch < 2:
                loss = loss
            else:
                 loss = args.beta * loss1 + (1. - args.beta) * contrastiveLoss
            # loss = args.beta * loss1 + (1. - args.beta) * contrastiveLoss

            if epoch >= 0:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

            for v in range(n_view):
                loss_list[v] += losses[v]
                _, predicted = preds[v].max(1)
                total_list[v] += targets[v].size(0)
                acc = predicted.eq(targets[v]).sum().item()
                correct_list[v] += acc
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

        train_dataset.testClean(select_idx)
        train_dict = {('view_%d_loss' % v): loss_list[v] / len(train_loader) for v in range(n_view)}
        train_dict['sum_loss'] = train_loss / len(train_loader)
        summary_writer.add_scalars('Loss/train', train_dict, epoch)
        summary_writer.add_scalars('Accuracy/train', {'view_%d_acc': correct_list[v] / total_list[v] for v in range(n_view)}, epoch)

    def eval(data_loader, epoch, mode='test'):
        fea, lab = [[] for _ in range(n_view)], [[] for _ in range(n_view)]
        test_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view
        with torch.no_grad():
            if sum([data_loader.dataset.train_data[v].shape[0] != data_loader.dataset.train_data[0].shape[0] for v in range(len(data_loader.dataset.train_data))]) == 0:
                for batch_idx, (batches, targets, index) in enumerate(data_loader):
                    batches, targets = [batches[v].cuda() for v in range(n_view)], [targets[v].cuda() for v in range(n_view)]
                    outputs = [multi_models[v](batches[v]) for v in range(n_view)]
                    pred, losses = [], []
                    for v in range(n_view):
                        fea[v].append(outputs[v])
                        lab[v].append(targets[v])
                        pred.append(outputs[v].mm(C))
                        losses.append(criterion(pred[v], targets[v]))
                        loss_list[v] += losses[v]
                        _, predicted = pred[v].max(1)
                        total_list[v] += targets[v].size(0)
                        acc = predicted.eq(targets[v]).sum().item()
                        correct_list[v] += acc
                    loss = sum(losses)
                    test_loss += loss.item()
            else:
                pred, losses = [], []
                for v in range(n_view):
                    count = int(np.ceil(data_loader.dataset.train_data[v].shape[0]) / data_loader.batch_size)
                    for ct in range(count):
                        batch, targets = torch.Tensor(data_loader.dataset.train_data[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).cuda(), torch.Tensor(data_loader.dataset.noise_label[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).long().cuda()
                        outputs = multi_models[v](batch)

                        fea[v].append(outputs)
                        lab[v].append(targets)
                        pred.append(outputs.mm(C))
                        losses.append(criterion(pred[v], targets))
                        loss_list[v] += losses[v]
                        _, predicted = pred[v].max(1)
                        total_list[v] += targets.size(0)
                        acc = predicted.eq(targets).sum().item()
                        correct_list[v] += acc
                    loss = sum(losses)
                    test_loss += loss.item()

            fea = [torch.cat(fea[v]).cpu().detach().numpy() for v in range(n_view)]
            lab = [torch.cat(lab[v]).cpu().detach().numpy() for v in range(n_view)]
        test_dict = {('view_%d_loss' % v): loss_list[v] / len(data_loader) for v in range(n_view)}
        test_dict['sum_loss'] = test_loss / len(data_loader)
        summary_writer.add_scalars('Loss/' + mode, test_dict, epoch)

        summary_writer.add_scalars('Accuracy/' + mode, {('view_%d_acc' % v): correct_list[v] / total_list[v] for v in range(n_view)}, epoch)
        return fea, lab

    def multiview_test(fea, lab):
        MAPs = np.zeros([n_view, n_view])
        val_dict = {}
        print_str = ''
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                key = '%s2%s' % (args.views[i], args.views[j])
                val_dict[key] = MAPs[i, j]
                print_str = print_str + key + ': %.3f\t' % val_dict[key]
        return val_dict, print_str

    def test(epoch, is_eval=True):
            global best_acc
            set_eval()
            # switch to evaluate mode
            # fea, lab = eval(train_loader, epoch, 'train')
            #
            # MAPs = np.zeros([n_view, n_view])
            # train_dict = {}
            # for i in range(n_view):
            #     for j in range(n_view):
            #         MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
            #         train_dict['%s2%s' % (args.views[i], args.views[j])] = MAPs[i, j]
            #
            # train_avg = MAPs.sum() / n_view / (n_view - 1.)
            # train_dict['avg'] = train_avg
            # summary_writer.add_scalars('Retrieval/train', train_dict, epoch)

            fea, lab = eval(valid_loader, epoch, 'valid')
            # if is_eval:
            #     fea = [fea[v][0: 2000] for v in range(n_view)]
            #     lab = [lab[v][0: 2000] for v in range(n_view)]

            MAPs = np.zeros([n_view, n_view])
            val_dict = {}
            test_dict = {}
            print_val_str = 'Validation: '
            print_test_str = 'Test: '

            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    val_dict[key] = MAPs[i, j]
                    print_val_str = print_val_str + key +': %g\t' % val_dict[key]

                    test_dict[key] = MAPs[i, j]
                    print_test_str = print_test_str + key + ': %g\t' % test_dict[key]


            val_avg = MAPs.sum() / n_view / (n_view - 1.)
            val_dict['avg'] = val_avg
            print_val_str = print_val_str + 'Avg: %g' % val_avg
            summary_writer.add_scalars('Retrieval/valid', val_dict, epoch)

            # fea, lab = eval(test_loader, epoch, 'test')
            # if is_eval:
            #     fea = [fea[v][0: 2000] for v in range(n_view)]
            #     lab = [lab[v][0: 2000] for v in range(n_view)]

            # MAPs = np.zeros([n_view, n_view])
            # test_dict = {}
            # print_test_str = 'Test: '
            # for i in range(n_view):
            #     for j in range(n_view):
            #         if i == j:
            #             continue
            #         MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
            #         key = '%s2%s' % (args.views[i], args.views[j])
            #         test_dict[key] = MAPs[i, j]
            #         print_test_str = print_test_str + key + ': %g\t' % test_dict[key]

            test_avg = MAPs.sum() / n_view / (n_view - 1.)
            print_test_str = print_test_str + 'Avg: %g' % test_avg
            test_dict['avg'] = test_avg
            summary_writer.add_scalars('Retrieval/test', test_dict, epoch)

            print(print_val_str)
            if val_avg > best_acc:
                best_acc = val_avg
                print(print_test_str)
                print('Saving..')
                state = {}
                for v in range(n_view):
                    # models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
                    state['model_state_dict_%d' % v] = multi_models[v].state_dict()
                for key in test_dict:
                    state[key] = test_dict[key]
                state['epoch'] = epoch
                state['optimizer_state_dict'] = optimizer.state_dict()
                state['C'] = C
                torch.save(state, os.path.join(args.ckpt_dir, '%s_%s_%d_best_checkpoint.t7' % ('MRL', args.data_name, args.output_dim)))
            return val_dict

    # test(1)
    best_prec1 = 0.
    lr_schedu.step(start_epoch)
    train(-1)
    results = test(-1)
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        test_dict = test(epoch + 1)
        if test_dict['avg'] == best_acc:
            multi_model_state_dict = [{key: value.clone() for (key, value) in m.state_dict().items()} for m in multi_models]
            W_best = C.clone()

    print('Evaluation on Last Epoch:')
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)

    print('Evaluation on Best Validation:')
    [multi_models[v].load_state_dict(multi_model_state_dict[v]) for v in range(n_view)]
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)
    import scipy.io as sio
    save_dict = dict(**{args.views[v]: fea[v] for v in range(n_view)}, **{args.views[v] + '_lab': lab[v] for v in range(n_view)})
    save_dict['C'] = W_best.detach().cpu().numpy()
    sio.savemat('features/%s_%g.mat' % (args.data_name, args.noisy_ratio), save_dict)
def mixgen_batch(data, targets, lam=0.5):
    image = data[0]
    text = data[1]
    target1 = targets[0]
    target2 = targets[0]
    size = image.size()[0]
    index = np.random.permutation(size)
    for i in range(size):
        # image mixup
        image[i, :] = lam * image[i, :] + (1 - lam) * image[index[i], :]
        # text concat
        text[i, :] = lam * text[i, :] + (1 - lam) * text[index[i], :]

        target1[i] = lam * target1[i] + (1 - lam) * target1[index[i]]
        target2[i] = lam * target1[i] + (1 - lam) * target1[index[i]]
    return [image, text], [target1, target2]
def fx_calc_map_multilabel_k(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(train_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)

    ord = dist.argsort(1)

    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

if __name__ == '__main__':
    main()
