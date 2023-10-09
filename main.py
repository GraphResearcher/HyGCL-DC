import time
from tqdm import tqdm
from argument import parse_args
from utils import *


def train(model, data, device, args, cl=False):
    if cl:
        out = model.forward_cl(data, data.structure)
    else:
        out = model.forward(data, data.structure)
    return out


def run(args):

    fix_seed(args.seed)
    logger = Logger(args.runs, args)

    if args.cuda in [0, 1, 2, 3]:
        device = torch.device('cuda:' + str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')


    data, args = load_data(args)
    data.device = device
    data.update()
    model = parse_model(args)
    model.to(device)
    activation, loss_function, cl_function, pred_function = get_functions(args.dname)


    split_idx_list = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(data.Y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_list.append(split_idx)

    runtime_list = []
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_list[run]
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(args.epochs):

            model.train()
            optimizer.zero_grad()

            data_aug1 = aug(dataset=data, method=args.aug_method, ratio=args.aug_ratio, deepcopy=True)
            data_aug2 = aug(dataset=data, method=args.aug_method, ratio=args.aug_ratio, deepcopy=True)

            out1 = train(model=model, data=data_aug1, device=device, args=args, cl=True)
            out2 = train(model=model, data=data_aug2, device=device, args=args, cl=True)

            cl_loss = cl_function(out1, out2, args.cl_temperature)

            loss = cl_loss * args.alpha

            output = train(model=model, data=data, device=device, args=args, cl=False)
            logit = activation(output) if activation == F.sigmoid else activation(output, dim=1)

            model_loss = loss_function(logit[train_idx], data.Y_TORCH[train_idx])
            loss += model_loss

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                output = train(model=model, data=data, device=device, args=args, cl=False)
                logit = activation(output) if activation == F.sigmoid else activation(output, dim=1)

                train_loss = loss_function(logit[train_idx], data.Y_TORCH[train_idx])
                valid_loss = loss_function(logit[valid_idx], data.Y_TORCH[valid_idx])
                test_loss = loss_function(logit[test_idx], data.Y_TORCH[test_idx])
                y_pred = pred_function(logit, args.threshold)

                result = evaluate(y_true=data.Y_TORCH, y_pred=y_pred, split_idx=split_idx, num_classes=args.num_classes,
                                  loss_function=loss_function)
                logger.add_result(run, result[:6])

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'CL Loss: {cl_loss:.4f}, '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Valid Loss: {valid_loss:.4f}, '
                      f'Test  Loss: {test_loss:.4f}, '
                      f'Train F1: {100 * result[0]:.2f}%, '
                      f'Train Jaccard: {100 * result[1]:.2f}%, '
                      f'Valid F1: {100 * result[2]:.2f}%, '
                      f'Valid Jaccard: {100 * result[3]:.2f}%, '
                      f'Test F1: {100 * result[4]:.2f}%, '
                      f'Test Jaccard: {100 * result[5]:.2f}%, '
                      )
        end_time = time.time()
        runtime_list.append(end_time - start_time)

    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val_f1, best_val_jaccard, best_test_f1, best_test_jaccard = logger.print_statistics()

    res_root = osp.join(args.root_dir, 'results')
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.method}_{args.lr}_{args.wd}'
        cur_line += f',{best_val_f1.mean():.3f} ± {best_val_f1.std():.3f}'
        cur_line += f',{best_val_jaccard.mean():.3f} ± {best_val_jaccard.std():.3f}'
        cur_line += f',{best_test_f1.mean():.3f} ± {best_test_f1.std():.3f}'
        cur_line += f',{best_test_jaccard.mean():.3f} ± {best_test_jaccard.std():.3f}'
        cur_line += f',{avg_time // 60}min{(avg_time % 60):.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')


if __name__ == '__main__':
    args = parse_args()
    run(args)