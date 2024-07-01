from moviepy.editor import VideoFileClip
import os
import tqdm

def convert_mpg_to_mp4(input_file, output_file):
  clip = VideoFileClip(input_file)
  clip.write_videofile(output_file)

for i in tqdm.tqdm(os.listdir('data/videos')):
  output_file = 'data/final_videos/' + i.split('.')[0] + '.mp4'
  convert_mpg_to_mp4(f'data/videos/{i}', output_file)