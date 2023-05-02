all_preds = []
for X in tqdm(test_dataloader):
    X = X.to(torch.device('cuda:1'))
    pred = model1(X)
    all_preds += list(pred.cpu().argmax(axis=1).numpy())

classes = [label_decodes[i] for i in all_preds]
df_sub['Expected'] = classes
df_sub.to_csv('submission5_justbase.csv',index=False)