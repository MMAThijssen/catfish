from sys import argv

def parse_txt(infile):
    """
    """
    training = []
    validation = []
    sizes = []
    
    new_round = True
    get_size = False
    with open(infile, "r") as source:
        for line in source:
            if new_round:
                if get_size:
                    size = int(line.strip())
                    get_size = False
                if line.startswith("Training accuracy"):
                    train_acc = float(line.strip().split(": ")[1])
                elif line.startswith("Validation accuracy"):
                    val_acc = float(line.strip().split(": ")[1])
                    new_round = False
                elif line.startswith("Training loss"):
                    get_size = True
            else:
                training.append(train_acc)
                validation.append(val_acc)
                sizes.append(size)
                new_round = True
                
    return training, validation, sizes
    
print(len(parse_txt(argv[1])[2]))
        
    