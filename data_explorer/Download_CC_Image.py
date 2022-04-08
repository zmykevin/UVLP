#Download the CC Images
import json
from tqdm import tqdm
import csv

from urllib.request import urlretrieve

if __name__ == "__main__":
	with open("/home/zmykevin/fb_intern/data/mingyang_data/CC/cc_clip_test.json", "r") as f:
		clip_test = json.load(f)

	#Load the original CC Annotation
	tsv_file = open("/home/zmykevin/fb_intern/data/mingyang_data/CC/Train_GCC-training.tsv")
	read_tsv = csv.reader(tsv_file, delimiter="\t")

	output_path = "/home/zmykevin/fb_intern/data/mingyang_data/CC/clip_score_img"
	caption_url = {}
	for row in read_tsv:
	  caption_url[row[0]] = row[1]

	invalid_count = 0
	for k,v in tqdm(clip_test.items()):
		current_caption = v["caption"]
		cc_id = v["cc_id"]
		if caption_url.get(current_caption, None) is not None:
			current_url = caption_url[current_caption]
			try:
				urlretrieve(current_url, "{}/{}.jpg".format(output_path, cc_id))
			except:
				invalid_count += 1
		else:
			invalid_count +=1

	print(invalid_count)
