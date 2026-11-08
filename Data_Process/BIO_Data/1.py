file = open("./BIO_Data/Agricultural_val_dataset_BIO1.txt", 'w', encoding='utf-8')

with open("./BIO_Data/Agricultural_val_dataset_BIO.txt", "r", encoding='utf-8') as f:
    for line in f:
        file.write(line)
        if line.strip() == "。 O":
            file.write("\n")