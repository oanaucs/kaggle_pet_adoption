import os

import json
import csv

import matplotlib.pyplot as plt


sent_files_dir = '/media/oanaucs/Data/kaggle_pet_adoption/train_sentiment/'
metadata_files_dir = '/media/oanaucs/Data/kaggle_pet_adoption/train_metadata/'

train_csv = '/media/oanaucs/Data/kaggle_pet_adoption/train.csv'

data_to_read = ['PetID', 'Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1',
                'Color2', 'Color3',
                'MaturitySize', 'FurLength', 'Vaccinated',
                'Vaccinated', 'Sterilized',
                'Health', 'Quantity', 'Fee', 'State']


def read_metadata_files(metadata_files_dir):
    metadata_files = os.listdir(metadata_files_dir)
    metadata = dict()
    for file in metadata_files[:1]:
        with open(os.path.join(metadata_files_dir, file)) as d:
            data = json.load(d)
            print(data)
            # metadata.update({pet_sent.split('.')[0]: (data['documentSentiment']['magnitude'], data['documentSentiment']['score'])})
    return metadata


def read_sentiment_files(sent_files_dir):
    sent_files = os.listdir(sent_files_dir)
    sent_data = dict()
    for file in sent_files:
        with open(os.path.join(sent_files_dir, file)) as d:
            data = json.load(d)
            sent_data.update({pet_sent.split('.')[0]: (
                data['documentSentiment']['magnitude'],
                data['documentSentiment']['score'])})
    return sent_data


def read_train_image_and_csv(train_csv_file, images_dir):
	train_data = dict()
    with open(train_csv_file) as file:
        reader = csv.DictReader(file)
        for row in reader:
            curr_id = row['PetID']
            # read rest info
			curr_image = plt.imread(os.path.join(image_dir, id))
            train_data.update(curr_id: curr_image)

    return train_data


def main():
    # sent_data = read_sentiment_files(sent_files_dir)
    # metadata = read_metadata_files(metadata_files_dir)
    read_train_csv(train_csv)


if __name__ == '__main__':
    main()
