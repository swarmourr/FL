from itertools import combinations

import glob


class Selection():


    def randomselection(self, data_path,client):
        files=glob.glob(data_path+'*')
        participants = combinations(files, client)
        return participants

