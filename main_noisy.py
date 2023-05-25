import numpy as np
import os
import torch
# import numpy as np
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import torch
from utils.config import args
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
import nets as models
from utils.bar_show import progress_bar
from src.noisydataset import cross_modal_dataset
from src.smoothCE import smoothCE
from src.bmm import BetaMixture1D
import src.utils as utils
import scipy
import scipy.spatial
import numpy.ma as ma
from sklearn.mixture import GaussianMixture as GMM
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer

save_dir = 'aum'
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
    C = torch.nn.init.orthogonal(C, gain=1)[:, 0: train_dataset.class_num]
    C.requires_grad = False

    embedding = torch.eye(train_dataset.class_num).cuda()
    embedding.requires_grad = False

    parameters = []
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
        criterion_no_mean = torch.nn.CrossEntropyLoss(reduction='none')
    elif args.loss == 'smoothCE':
        criterion = smoothCE(0.1, train_dataset.class_num)
        criterion_no_mean = smoothCE(0.1, train_dataset.class_num, 1)
    elif args.loss == 'MCE':
        criterion = utils.MeanClusteringError(train_dataset.class_num, tau=args.tau).cuda()
    else:
        raise Exception('No such loss function.')

    summary_writer = SummaryWriter(args.log_dir)

    def entropyLoss(a,tar): # 两个向量的交叉熵
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        res = -tar * logsoftmax(a)
        return torch.sum(res, dim=1)

    CE = torch.nn.CrossEntropyLoss().cuda()
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

    def contrastive(fea, tar, tau=1.):
        loss = []
        for v in range(n_view):
            sim = fea[v].mm(fea[v].t())
            sim = (sim / tau).exp()
            dif = tar[v]-tar[v].reshape(-1, 1)
            condition1 = dif == 0
            condition2 = dif != 0
            masked_sim1 = torch.masked_fill(sim, condition1, value=0)  # not seem class
            masked_sim2 = torch.masked_fill(sim, condition2, value=0) # seem class
            top_value, top_i = torch.topk(masked_sim2, 1)
            select_sim = (top_value.sum()).reshape(1, -1).squeeze()
            loss.append(-(select_sim / masked_sim1.sum(1)).log())
        return torch.cat(loss)
    def cross_modal_contrastive_ctriterion(fea, tar, tau=1.):
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())
        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()

        all_targets = torch.cat(tar)
        re_targets = all_targets.reshape((-1, 1))
        dif = all_targets - re_targets
        condition1 = dif == 0
        masked_sim1 = torch.masked_fill(sim, condition1, value=0) #not seem class

        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss1 = -(diag1 / masked_sim1.sum(1)).log()
        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss2 = -(diag2 / masked_sim1.sum(1)).log()
        return loss1 + loss2
    def gmm_loss(x, pred):
        xx = x.cpu().detach().numpy()
        gmm = GMM(n_components=train_dataset.class_num, max_iter=10, tol=1e-2, reg_covar=5e-4).fit(xx)
        probs = gmm.predict_proba(xx)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        res = -torch.from_numpy(probs).cuda() * logsoftmax(pred)
        return torch.sum(res, dim=1)

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

            c = list()
            for cla in range(train_dataset.class_num):
                select_list = list()
                for v in range(n_view):
                    idx = torch.nonzero(targets[v] == cla).ravel()
                    select_list.append(torch.index_select(outputs[v], dim=0, index=idx))
                selects = torch.cat(select_list)
                if (len(selects)):
                    val = (torch.sum(selects, dim=0) / len(selects))
                else:
                    val = C[..., cla].cuda().t()
                c.append(val)
            t = torch.stack(c)
            C.data = t.t()

            preds = [outputs[v].mm(C) for v in range(n_view)]
            s_CE_loss = [criterion_no_mean(preds[v], targets[v]) for v in range(n_view)]
            losses = [torch.mean(s_CE_loss[v]) for v in range(n_view)]
            s_CE_loss = torch.stack(s_CE_loss).reshape(1, -1).squeeze()

            # ce_f  = torch.nn.CrossEntropyLoss(reduction='none')
            # ce_loss = [ce_f(preds[v], targets[v]) for v in range(n_view)]
            # ce_loss = torch.stack(ce_loss).reshape(1, -1).squeeze()

            loss_nor = (s_CE_loss - s_CE_loss.min()) / (s_CE_loss.max() - s_CE_loss.min())
            # loss_nor = (ce_loss - ce_loss.min()) / (ce_loss.max() - ce_loss.min())
            bmm_A = BetaMixture1D(max_iters=10)
            loss_i = loss_nor.reshape(-1, 1)[0:batch_size].cpu().detach().numpy()
            bmm_A.fit(loss_i)
            prob_A = bmm_A.posterior(loss_i, 0).T.squeeze()

            bmm_B = BetaMixture1D(max_iters=10)
            loss_t = loss_nor.reshape(-1, 1)[batch_size:].cpu().detach().numpy()
            bmm_B.fit(loss_t)
            prob_B = bmm_A.posterior(loss_t, 0).T.squeeze()

            select_num = len(prob_A) // 10
            threshld_A = np.sort(prob_A)[::-1][select_num]
            threshld_B = np.sort(prob_B)[::-1][select_num]
            # if epoch>3:
            pred_A = (prob_A > threshld_A).squeeze()
            pred_B = (prob_B > threshld_B).squeeze()

            # pred = np.hstack((pred_A, pred_B)).T.squeeze()
            selected = [pred_A, pred_B]
            inputs_x =  [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            targets_x = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            inputs_u = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            targets_u = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            outputs_u = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            all_inputs = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]

            classes = torch.arange(0, train_dataset.class_num, 1).cuda()
            added = [torch.cat([targets[v], classes], dim=0) for v in range(n_view)]
            all_targets = [LabelBinarizer().fit_transform(added[v].cpu().detach().numpy())[:-train_dataset.class_num] for v in range(n_view)]

            # all_targets = [LabelBinarizer().fit_transform(targets[v].cpu().detach().numpy()) for v in range(n_view)]
            for v in range(n_view):
                inputs_x[v] = batches[v][selected[v]]
                targets_x[v] = all_targets[v][selected[v]]
                inputs_u[v] = batches[v][~selected[v]]
                outputs_u[v] = preds[v][~selected[v]]
                targets_u[v] = all_targets[v][~selected[v]]

                all_inputs[v] = torch.cat([inputs_x[v], inputs_u[v]], dim=0).cuda()
                targets_x[v] = torch.from_numpy(targets_x[v])
                targets_u[v] = torch.from_numpy(targets_u[v])
                all_targets[v] = torch.cat([targets_x[v], targets_u[v]], dim=0).cuda()

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                for v in range(n_view):
                    p = torch.softmax(outputs_u[v], dim=1)
                    pt = p ** (1 / args.T)
                    targets_u[v] = pt / pt.sum(dim=1, keepdim=True)
                    targets_u[v] = targets_u[v].detach()
            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs[0].size(0))
            mixed_input = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            mixed_target = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            logits = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            Lx = [torch.tensor([], dtype=torch.float64) for i in range(n_view)]
            prior = torch.ones(train_dataset.class_num) / train_dataset.class_num
            prior = prior.cuda()
            for v in range(n_view):
                mixed_input[v] = l * all_inputs[v] + (1 - l) * all_inputs[v][idx]
                mixed_target[v] = l * all_targets[v] + (1 - l) * all_targets[v][idx]
                logits[v] = multi_models[v](mixed_input[v]).mm(C)

                # pred_mean = torch.softmax(logits[v], dim=1).mean(0)
                # penalty = torch.sum(prior * torch.log(prior / pred_mean))

                Lx[v] = entropyLoss(logits[v],mixed_target[v]) # + penalty

            lx_loss = torch.cat(Lx)

            for v in range(n_view):
                ss = index[selected[v]]
                select_idx[v] = torch.cat([select_idx[v], ss], dim=0)


            contrastiveLoss = 0.05*contrastive(outputs, targets, tau=args.tau) + cross_modal_contrastive_ctriterion(outputs, targets, tau=args.tau)
            # contrastiveLoss = cross_modal_contrastive_ctriterion(outputs, targets, tau=args.tau)
            # if epoch < 10:
            #     loss_all = 1 * s_CE_loss+ 1 * contrastiveLoss
            # else:
            loss_all = 0.7 * s_CE_loss+ 1 * contrastiveLoss + 0.3 * lx_loss
            ind_sorted = np.argsort(loss_all.cpu().detach().numpy())
            loss_sorted = loss_all[ind_sorted]
            remember_rate = 1
            # remember_rate = 1 - min((epoch + 2) / 10 * args.noisy_ratio, args.noisy_ratio)
            # if epoch < 5:
            #     remember_rate = 1
            num_remember = int(remember_rate * len(loss_sorted))
            ind_update = ind_sorted[:num_remember]
            loss = torch.mean(loss_all[ind_update])
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
            if is_eval:
                fea = [fea[v][0: 2000] for v in range(n_view)]
                lab = [lab[v][0: 2000] for v in range(n_view)]
            MAPs = np.zeros([n_view, n_view])
            val_dict = {}
            print_val_str = 'Validation: '

            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    val_dict[key] = MAPs[i, j]
                    print_val_str = print_val_str + key +': %g\t' % val_dict[key]


            val_avg = MAPs.sum() / n_view / (n_view - 1.)
            val_dict['avg'] = val_avg
            print_val_str = print_val_str + 'Avg: %g' % val_avg
            summary_writer.add_scalars('Retrieval/valid', val_dict, epoch)

            fea, lab = eval(test_loader, epoch, 'test')
            if is_eval:
                fea = [fea[v][0: 2000] for v in range(n_view)]
                lab = [lab[v][0: 2000] for v in range(n_view)]
            MAPs = np.zeros([n_view, n_view])
            test_dict = {}
            print_test_str = 'Test: '
            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    test_dict[key] = MAPs[i, j]
                    print_test_str = print_test_str + key + ': %g\t' % test_dict[key]

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

    # print('Evaluation on Last Epoch:')
    # fea, lab = eval(test_loader, epoch, 'test')
    # test_dict, print_str = multiview_test(fea, lab)
    # print(print_str)

    print('Evaluation on Best Validation:')
    [multi_models[v].load_state_dict(multi_model_state_dict[v]) for v in range(n_view)]
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)
    import scipy.io as sio
    save_dict = dict(**{args.views[v]: fea[v] for v in range(n_view)}, **{args.views[v] + '_lab': lab[v] for v in range(n_view)})
    save_dict['C'] = W_best.detach().cpu().numpy()
    sio.savemat('features/%s_%g.mat' % (args.data_name, args.noisy_ratio), save_dict)

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

