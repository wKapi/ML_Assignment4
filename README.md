# Challenges in Representation Learning: Facial Expression Recognition Challenge
## კონკურსის მიმოხილვა
კონკურსის მიზანია ადამიანის სახის სურათების მიხედვით 7 ძირითადი ემოციის პროგნოზირება. შეფასება ხდება Accuracy (სიზუსტის) მეტრიკით.

## რეპოზიტორიის სტრუქტურა
```
Facial-Expression-Recognition/
│
├── FER2013_Notebook.ipynb    ← Data Loading, Augmentation, All 4 Models, Training Loop, W&B Logging
└── README.md                 ← ექსპერიმენტების ანალიზი და შეჯამება
```
---
## ფაილების აღწერა

| ფაილი | აღწერა |
|---|---|
| `notebook.ipynb`| მონაცემების Kaggle-იდან ჩამოტვირთვა, პაიპლაინის აწყობა, 10 ექსპერიმენტის ჩატარება და საუკეთესო FaceResNet მოდელის ტრენინგი. |
| `README.md` | პროექტის დოკუმენტაცია და მიღებული შედეგების ანალიზი. |

---

## მონაცემთა გაწმენდა და დამუშავება (Pipeline & Preprocessing)

### 1. მონაცემების სტრუქტურიზაცია
* Kaggle-იდან გადმოწერილი პიქსელების სტრიქონები (pixels) გადავიყვანე NumPy მასივებად, შევცვალე მათი ფორმა 48×48 გარჩევადობაზე და შევქმენი
  FERDataset კლასი PyTorch-ისთვის.
### 2. ბაზისური ტრანსფორმაცია
* სურათები გადავიყვანე ტენზორებში (ToTensor) და მოვახდინე მათი ნორმალიზაცია საშუალო [0.5] და სტანდარტული გადახრა [0.5] პარამეტრებით, რათა პიქსელების მნიშვნელობები მოქცეულიყო [-1, 1] დიაპაზონში. მონაცემები გაიყო პროპორციით: 80% Train და 20% Validation.

---

## Feature Engineering
ბაზისურმა მოდელებმა აჩვენეს, რომ ქსელი სწრაფად იწყებდა Overfitting-ს. ამის გამოსასწორებლად სატრენინგო მონაცემებზე ჩავრთე მონაცემთა ხელოვნური გაძლიერება (Data Augmentation):
* RandomHorizontalFlip — სახის ჰორიზონტალური მიტრიალება;
* RandomRotation(15) — სახის მცირე დახრა (15 გრადუსით).
ამ მიდგომამ ხელოვნურად გაამრავლფეროვნა Dataset-ი და აიძულა მოდელი, რომ ზეპირად სწავლის ნაცვლად რეალური პატერნები დაეჭირა.

---

## ტრენინგი და ექსპერიმენტები

გამოვიყენე ოთხი სხვადასხვა არქიტექტურა: SmallCNN, DeepCNN, OptimalCNN და FaceResNet. პირველი სამი მოდელისთვის ჩავატარე 3-3 ექსპერიმენტი (Baseline, High LR, Small Batch + Weight Decay) 10 ეპოქაზე, ხოლო ბოლო მოდელი გაიწვრთნა 30 ეპოქაზე ReduceLROnPlateau schedulerit.

### SmallCNN
* Baseline (LR=0.001, BS=64): Train Acc ≈ 42%, Val Acc ≈ 41%. ქსელი ზედმეტად პატარაა და არის Underfitting-ის პრობლემა.
* High_LR (LR=0.01): მივიღეთ ძაან დაბალი სიზუსტე (~18%). დიდი ნაბიჯის გამო ოპტიმიზატორი ვერ პოულობს მინიმუმს.
* Small_Batch_WD: მცირე ბეტჩმა და რეგულარიზაციამ შედეგი მხოლოდ 44%-მდე ასწია.
### DeepCNN
* Baseline (LR=0.001, BS=64): Train Acc ≈ 68%, Val Acc ≈ 53%. აშკარაა Overfitting-ის პრობლემა (სატრენინგო სიზუსტე იზრდება, მაგრამ ვალიდაცია ფუჭდება).
* High_LR (LR=0.01): ქსელი საერთოდ ვერ სწავლობს (Divergence, ~14%).
* Small_Batch_WD: Weight Decay-მ ცოტათი კი შეამცირა Overfitting, თუმცა ვალიდაცია მაინც 54%-ზე გაჩერდა.

### OptimalCNN(Batchnorm + Dropout)
* Baseline (LR=0.001, BS=64): Train Acc ≈ 56%, Val Acc ≈ 55%. Dropout-მა და BatchNorm-მა წაშალა Overfitting, თუმცა 10 ეპოქა არ აღმოჩნდა საკმარისი მაღალი სიზუსტისთვის.
* Small_Batch_WD: მივიღეთ 58% Train Acc და 56% Val Acc, რაც საუკეთესო შედეგია ბაზისურ მოდელებში.

### FaceResNet (Modern ResNet-like Architecture)
ეს მოდელი გავაკეთე Residual Blocks (Skip Connections) არქიტექტურით და გავუშვი გაძლიერებულ მონაცემებზე 30 ეპოქაზე:
* მიღებული შედეგი: Train Acc = 78.82% (Loss: 0.58) | Val Acc = 64.35% (Loss: 1.08).
* ანალიზი: Skip Connection-ების წყალობით გრადიენტებმა სიღმეში კარგად იმუშავა. მოდელმა აჩვენა საუკეთესო შედეგი. ბოლო ეპოქებზე შეინიშნება მცირე Overfitting, რის გამოსასწორებლადაც მომავალში საჭირო იქნება Dropoutის პროცენტის კიდევ უფრო გზრდა.

## საუკეთესო მოდელი
საუკეთესო მოდელად ერთმნიშვნელოვნად შეირჩა FaceResNet, რადგან:
* გამოიყენება მონაცემთა გაძლიერება (Augmentation), რაც ზრდის გენერალიზაციას;
* Skip Connections სტრუქტურა უკეთ იჭერს სახის რთულ გამომეტყველებებს;
* აჩვენა ყველაზე მაღალი Validation Accuracy (64.35%) და ყველაზე დაბალი Loss.

## Weights & Biases (W&B) Tracking
ყველა ჩატარებული ექსპერიმენტი და რანი დარეგისტრირებულია Weights & Biases პლატფორმაზე:
*  **W&B პროექტის გვერდი:** (https://wandb.ai/mkapa22-free-university-of-tbilisi-/facial-expression-recognition?nw=nwusermkapa22)
თითოეულ რანში დაილოგა:
* **ჰიპერპარამეტრები:** model_architecture, learning_rate, batch_size, weight_decay, epochs;
* **მეტრიკები:** train_loss, train_acc, val_loss, val_acc და lr (ცვლილება დროში სქედულერის მიერ).
