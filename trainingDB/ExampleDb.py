import ZODB, ZODB.FileStorage, BTrees.IOBTree
from os.path import isfile
import random

class ExampleDb(object):
    """
    A class for a database storing training examples for a neural network
    """
    def __init__(self, **kwargs):

        self._db = None
        self.nb_pos = 0
        self.nb_neg = 0
        if not isfile(kwargs['db_name']):
            self.width = kwargs['width']
        self.db_name = kwargs['db_name']
        self.db = self.db_name

    # get neg and pos from a read added for training
    def add_training_read(self, training_read):
        with self._db.transaction() as conn:     # start a transaction
            pos_examples, pos_labels = training_read.get_pos(self.width)
            for i, ex in enumerate(pos_examples):
                #TODO: check on nb_pos + i and pos_labels[i]
                conn.root.pos[self.nb_pos+i] = (ex, pos_labels[i]) # saves to transaction as (index: example)
            self.nb_pos += len(pos_examples)
            # TODO: how many samples to add? - 5 times was arbitrarily chosen: len(pos_examples) * 5 - now equal because more pos
            neg_examples, neg_labels = training_read.get_neg(self.width, len(pos_examples))
            for i, ex in enumerate(neg_examples):
                conn.root.neg[self.nb_neg+i] = (ex, neg_labels[i])
            self.nb_neg += len(neg_examples)         
                
                
    def get_training_set(self, size, sets="train"):
        """
        Return a balanced subset of reads from the DB
        :param size: number of reads to return
        :param includes: k-mers that should forcefully be included, if available in db
        :return: lists of numpy arrays for training data(x_out) and labels (y_out)
        """
        if sets == "train":
            nb_pos = size // 2
            nb_neg = size - nb_pos
        elif sets == "test":
            content = 0.0222269808106684 
            nb_pos = round(size * content)
            nb_neg = size - nb_pos

#        nb_pos = size // 2
#        nb_neg = size - nb_pos            

        ps = random.sample(range(self.nb_pos), nb_pos)
        ns = random.sample(range(self.nb_neg), nb_neg)
        
        print("Getting examples now..")

        with self._db.transaction() as conn:
            examples_pos = [conn.root.pos[n] for n in ps]   # conn.root.pos[n] is tuple(arrays, labels)
            print("Finished retrieving positives.")            
            examples_neg = [conn.root.neg[n] for n in ns]
        data_out = examples_pos + examples_neg
        random.shuffle(data_out)
        x_out, y_out = zip(*data_out)
        print("Calculating percentage HPs.")
        pos_count = 0
        neg_count = 0
        for y in y_out:
            pos = y.count(1)
            neg = y.count(0)
            pos_count += pos
            neg_count += neg
        t_count = len(y_out) * 35
        print("Pos count: {}\ntotal count: {}\nPercentage: {}\nNeg count: {}".format(pos_count, t_count, pos_count/t_count*100, neg_count))             
        return x_out, y_out

    def pack_db(self):
        self._db.pack()

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, db_name):
        """
        Construct ZODB database if not existing, store DB object
        :param db_name: name of new db, including path
        """
        is_existing_db = isfile(db_name)
        if is_existing_db:
            print("Opening db {db_name}".format(db_name=db_name))
        storage = ZODB.FileStorage.FileStorage(db_name)
        self._db = ZODB.DB(storage)
        if is_existing_db:
            with self._db.transaction() as conn:
                self.width = conn.root.width
                self.nb_pos = len(conn.root.pos)
                self.nb_neg = len(conn.root.neg)

        else:
            with self._db.transaction() as conn:
                conn.root.width = self.width
                conn.root.pos = BTrees.IOBTree.BTree()
                conn.root.neg = BTrees.IOBTree.BTree()
