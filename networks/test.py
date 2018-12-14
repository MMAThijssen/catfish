#list_a = [[0], [0.002, 0.004, 0.006, 0.009, 0.011, 0.014, 0.016, 0.018, 0.021, 0.024, 0.026, 0.029, 0.032, 0.034, 0.037], [0.001, 0.002, 0.003, 0.004, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.013, 0.014, 0.016, 0.017, 0.019, 0.021, 0.023, 0.025, 0.026, 0.028, 0.03, 0.031, 0.033, 0.035, 0.037, 0.039, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05, 0.052, 0.054, 0.056, 0.057, 0.059, 0.061, 0.063, 0.065, 0.067, 0.069, 0.071, 0.073, 0.075, 0.077, 0.079, 0.081, 0.082, 0.084, 0.086, 0.088, 0.09, 0.092, 0.094, 0.096, 0.098, 0.1, 0.102, 0.104, 0.106, 0.108], [0.003, 0.007, 0.011], [0.001, 0.003, 0.005, 0.007, 0.008, 0.01, 0.013, 0.015, 0.017, 0.019, 0.021, 0.023, 0.026, 0.028, 0.03, 0.032, 0.035, 0.037, 0.039, 0.041, 0.044, 0.046, 0.048, 0.05, 0.053, 0.055, 0.057, 0.06, 0.062, 0.064, 0.066, 0.069, 0.071, 0.073, 0.076, 0.078, 0.08, 0.083, 0.085, 0.088, 0.09, 0.092, 0.095, 0.097, 0.099, 0.102, 0.104, 0.107, 0.109, 0.111, 0.114, 0.116, 0.119, 0.121, 0.123, 0.126, 0.128, 0.131, 0.133, 0.135]]
#list_b = [[0], [102400, 204800, 307200, 409600, 512000, 614400, 716800, 819200, 921600, 1024000, 1126400, 1228800, 1331200, 1433600, 1536000], [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624, 27648, 28672, 29696, 30720, 31744, 32768, 33792, 34816, 35840, 36864, 37888, 38912, 39936, 40960, 41984, 43008, 44032, 45056, 46080, 47104, 48128, 49152, 50176, 51200, 52224, 53248, 54272, 55296, 56320, 57344, 58368, 59392, 60416, 61440, 62464, 63488], [5120000, 10240000, 15360000], [10240, 20480, 30720, 40960, 51200, 61440, 71680, 81920, 92160, 102400, 112640, 122880, 133120, 143360, 153600, 163840, 174080, 184320, 194560, 204800, 215040, 225280, 235520, 245760, 256000, 266240, 276480, 286720, 296960, 307200, 317440, 327680, 337920, 348160, 358400, 368640, 378880, 389120, 399360, 409600, 419840, 430080, 440320, 450560, 460800, 471040, 481280, 491520, 501760, 512000, 522240, 532480, 542720, 552960, 563200, 573440, 583680, 593920, 604160, 614400]]
#
#alist = []
#blist = []
#[alist.extend(lst) for lst in list_a]
#[blist.extend(lst) for lst in list_b]
##for lst in range(len(list_a)):
##    print(len(list_a[lst]), len((list_b[lst])))
##
#print(len(alist))
#print(len(alist) == len(blist))
#ab_list = zip(blist, alist)
#sorted_list = sorted(ab_list, key=lambda x: x[0])
#print(sorted_list)

number_list = list(range(9,21))
number_list.extend(list(range(22, 41)))
number_list.extend(list(range(42, 46)))
number_list.extend(list(range(47, 53)))

print(number_list)

db_dir_val_list = ["biGRU-RNN_{}".format(i) for i in number_list]
print(db_dir_val_list)

["biGRU-RNN_{}".format(i) for i in number_list]

    #0. Get input
    if not len(argv) == 7:
        raise ValueError("The following arguments should be provided in this order:\n" + 
                         "\t-network type\n\t-model id\n\t-path to training db" +
                         "\n\t-number of training reads\n\t-number of epochs" + 
                         "\n\t-path to validation db\n\t-max length of validation reads")
    
    number_list = list(range(9,21))
    number_list.extend(list(range(22, 41)))
    number_list.extend(list(range(42, 46)))
    number_list.extend(list(range(47, 53)))
    
    network_type = argv[1]
    db_dir_val = argv[2]
    max_seq_length = int(argv[3])                  
    
    # Keep track of memory and time
    p = psutil.Process(os.getpid())
    t1 = datetime.datetime.now() 
    m1 = p.memory_full_info().pss
    print("\nMemory use at start is", m1)  
    print("Started script at {}\n".format(t1))
        
    # 1. Restore model
    hpm_dict = retrieve_random_hyperparameters(network_type)
    model = build_model(network_type, **hpm_dict)
    model.restore_network()
    t2 = datetime.datetime.now()  
    m2 = p.memory_full_info().pss
    print("\nMemory after building model is ", m2)
    print("Built and initialized model in {}\n".format(t2 - t1))

    
    #3. Assess performance on validation set
    print("Loading validation database..")
    squiggles = helper_functions.load_squiggles(db_dir_val)
    validate(model, squiggles, max_seq_length)
    t4 = datetime.datetime.now()  
    m4 = p.memory_full_info().pss
    print("Memory use at end is ", m4)
    print("Validated model in {}".format(t4 - t3))