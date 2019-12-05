from package.dataset.data_cmt import *


class PCYC_dataloader(CMT_dataloader):
    def __init__(self, folder_sk=None, folder_im=None, clss=None, normalize01=False, doaug=False,
                 folder_nps=None, logger=None, sz=None, names=None, paired=False):
        """
        Attirbute:
            mode: IM/SK. Indicating it is image or sketch to be retrieved. Default: IM.
        :param folder_sk: sketch folder
        :param folder_im: image folder
        :param clss: classes to load
        :param normalize01: whether normalize data to 0-1
        :param doaug: whether do data augmentation
        :param folder_nps: the folder saves npy files. This allow fewer inodes to save the datasets(the server
                    does not allow too many inodes allocated). The folder should contain
                            classname1_sk.npy, classname1_im.npy,
                            classname2_sk.npy, classname1_im.npy,
                            ...
                    1. If folder_nps is None, folder_sk and folder_imsk must be provided.
                    2. If folder_nps is not None but no files exist in the folder, folder_sk and folder_im must be
                        provided, and such files would be created in folder_nps.
                    3. If folder_nps is not None and files exist in the folder, load the files instead of those
                        in folder_sk and folder_imsk for training.
        :param logger: logger to debug.
        :param sz: resize or not.
        :param names: the clsname_what_idx_2_imfilename.pkl file. Or a dict.
                clsname_what_idx_2_imfilename[class_name][im/st/sk][index] = image filename without postfix.
                Neccessary if data are paired.
        :param paired: paired data or not
        """
        super(PCYC_dataloader, self).__init__(
            folder_sk=folder_sk, folder_im=folder_im, clss=clss, normalize01=normalize01, doaug=doaug,
            folder_nps=folder_nps, logger=logger, sz=sz, names=names, paired=paired)

    def __getitem__(self, idx):
        cls_sk, idx_sk = self.idx2cls_items[SK][idx]
        idx_im = idx_sk if self.paired else np.random.randint(0, len(self.idx2skim_pair[cls_sk][IM]))
        return self.trans[SK](self.idx2skim_pair[cls_sk][SK][idx_sk]), self.trans[IM](
            self.idx2skim_pair[cls_sk][IM][idx_im]), np.int16(cls_sk)

