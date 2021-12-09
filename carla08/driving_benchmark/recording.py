import csv
import datetime
import os
import cv2
import json
import numpy as np
from google.protobuf.json_format import MessageToJson, MessageToDict
from ..image_converter import to_rgb_array


class Recording(object):

    def __init__(self,
                 name_to_save,
                 continue_experiment,
                 save_images,
                 save_videos,
                 save_processed_videos,
                 policy_roll_out
                 ):

        self._dict_summary = {'exp_id': -1,
                              'rep': -1,
                              'weather': -1,
                              'start_point': -1,
                              'end_point': -1,
                              'result': -1,
                              'initial_distance': -1,
                              'final_distance': -1,
                              'final_time': -1,
                              'time_out': -1,
                              'end_pedestrian_collision': -1,
                              'end_vehicle_collision': -1,
                              'end_other_collision': -1,
                              'number_red_lights': -1,
                              'number_green_lights': -1,
                              'initial_path_distance': -1
                              }
        self._dict_measurements = {'exp_id': -1,
                                   'rep': -1,
                                   'weather': -1,
                                   'start_point': -1,
                                   'end_point': -1,
                                   'collision_other': -1,
                                   'collision_pedestrians': -1,
                                   'collision_vehicles': -1,
                                   'intersection_otherlane': -1,
                                   'intersection_offroad': -1,
                                   'pos_x': -1,
                                   'pos_y': -1,
                                   'steer': -1,
                                   'throttle': -1,
                                   'brake': -1
                                   }

        # Just in the case is the first time and there is no benchmark results folder
        if not os.path.exists('_benchmarks_results'):
            os.mkdir('_benchmarks_results')

        # Generate the full path for the log files
        self._path = os.path.join('_benchmarks_results'
                                  , name_to_save
                                  )

        # Check for continuation of experiment, also returns the last line, used for test purposes
        # If you don't want to continue it will create a new path name with a number.
        # Also returns the fieldnames for both measurements and summary, so you can keep the
        # previous order
        self._path, _, self._summary_fieldnames, self._measurements_fieldnames\
            = self._continue_experiment(continue_experiment)

        self._create_log_files()

        # A log with a date file: to show when was the last access and log what was tested,
        now = datetime.datetime.now()
        self._internal_log_name = os.path.join(self._path, 'log_' + now.strftime("%Y%m%d%H%M"))
        open(self._internal_log_name, 'w').close()

        # store the save images flag, and already store the format for image saving
        self._save_images = save_images
        self._image_filename_format = os.path.join(
            self._path, '_images/episode_{:s}/{:s}/image_{:0>5d}.jpg')

        # store the videos flag, and store the format for video saving
        self._save_videos = save_videos
        self._video_filename_format = os.path.join(
            self._path, '_videos/episode_{:s}.mp4')
        self.video_writer = None

        # store the processed videos flag, and store the format for video saving
        self._save_processed_videos = save_processed_videos
        self._processed_video_filename_format = os.path.join(
            self._path, '_processed_videos/episode_{:s}.mp4')
        self.processed_video_writer = None

        # store the measurements and
        self._policy_roll_out = policy_roll_out
        self._measurements_filename_format = os.path.join(self._path,
                                                          '_measurements/episode_{:s}/measurements_{:0>5d}.json')
        self._control_filename_format = os.path.join(self._path,
                                                     '_control/episode_{:s}/control_{:0>5d}.json')

    @property
    def path(self):
        return self._path

    def log_poses(self, start_index, end_index, weather_id, episode_name):
        with open(self._internal_log_name, 'a+') as log:
            log.write(' Start Poses  (%d  %d ) on weather %d \n ' %
                      (start_index, end_index, weather_id))
        if self._save_videos:
            folder = os.path.dirname(self._video_filename_format.format(episode_name))
            if not os.path.isdir(folder):
                os.makedirs(folder)
            if os.path.exists(self._video_filename_format.format(episode_name)):
                os.remove(self._video_filename_format.format(episode_name))
            self.video_writer = cv2.VideoWriter(self._video_filename_format.format(episode_name),
                                                cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 600))
        if self._save_processed_videos:
            folder = os.path.dirname(self._processed_video_filename_format.format(episode_name))
            if not os.path.isdir(folder):
                os.makedirs(folder)
            if os.path.exists(self._processed_video_filename_format.format(episode_name)):
                os.remove(self._processed_video_filename_format.format(episode_name))
            self.processed_video_writer = cv2.VideoWriter(self._processed_video_filename_format.format(episode_name),
                                                          cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 600))

    def log_poses_finish(self):
        with open(self._internal_log_name, 'a+') as log:
            log.write('Finished Task')
        if self._save_videos:
            self.video_writer.release()
            self.video_writer = None
        if self._save_processed_videos:
            self.processed_video_writer.release()
            self.processed_video_writer = None

    def log_start(self, id_experiment):

        with open(self._internal_log_name, 'a+') as log:
            log.write('Start Task %s \n' % str(id_experiment))


    def log_end(self):
        with open(self._internal_log_name, 'a+') as log:
            log.write('====== Finished Entire Benchmark ======')

    def write_summary_results(self, experiment, pose, rep,
                              path_distance, remaining_distance,
                              final_time, time_out, result,
                              end_pedestrian, end_vehicle, end_other,
                              number_red_lights, number_green_lights, initial_path_distance):
        """
        Method to record the summary of an episode(pose) execution
        """

        self._dict_summary['exp_id'] = experiment.task
        self._dict_summary['rep'] = rep
        self._dict_summary['weather'] = experiment.Conditions.WeatherId
        self._dict_summary['start_point'] = pose[0]
        self._dict_summary['end_point'] = pose[1]
        self._dict_summary['result'] = result
        self._dict_summary['initial_distance'] = path_distance
        self._dict_summary['final_distance'] = remaining_distance
        self._dict_summary['final_time'] = final_time
        self._dict_summary['time_out'] = time_out
        self._dict_summary['end_pedestrian_collision'] = end_pedestrian
        self._dict_summary['end_vehicle_collision'] = end_vehicle
        self._dict_summary['end_other_collision'] = end_other
        self._dict_summary['number_red_lights'] = number_red_lights
        self._dict_summary['number_green_lights'] = number_green_lights
        self._dict_summary['initial_path_distance'] = initial_path_distance

        with open(os.path.join(self._path, 'summary.csv'), 'a+') as ofd:
            w = csv.DictWriter(ofd, self._dict_summary.keys())
            w.fieldnames = self._summary_fieldnames

            w.writerow(self._dict_summary)

    def write_measurements_results(self, experiment, rep, pose, reward_vec, control_vec):
        """
        Method to record the measurements, sensors,
        controls and status of the entire benchmark.
        """
        with open(os.path.join(self._path, 'measurements.csv'), 'a+') as rfd:
            mw = csv.DictWriter(rfd, self._dict_measurements.keys())
            mw.fieldnames = self._measurements_fieldnames
            for i in range(len(reward_vec)):
                self._dict_measurements['exp_id'] = experiment.task
                self._dict_measurements['rep'] = rep
                self._dict_measurements['start_point'] = pose[0]
                self._dict_measurements['end_point'] = pose[1]
                self._dict_measurements['weather'] = experiment.Conditions.WeatherId
                self._dict_measurements['collision_other'] = reward_vec[
                    i].collision_other
                self._dict_measurements['collision_pedestrians'] = reward_vec[
                    i].collision_pedestrians
                self._dict_measurements['collision_vehicles'] = reward_vec[
                    i].collision_vehicles
                self._dict_measurements['intersection_otherlane'] = reward_vec[
                    i].intersection_otherlane
                self._dict_measurements['intersection_offroad'] = reward_vec[
                    i].intersection_offroad
                self._dict_measurements['pos_x'] = reward_vec[
                    i].transform.location.x
                self._dict_measurements['pos_y'] = reward_vec[
                    i].transform.location.y
                self._dict_measurements['steer'] = control_vec[
                    i].steer
                self._dict_measurements['throttle'] = control_vec[
                    i].throttle
                self._dict_measurements['brake'] = control_vec[
                    i].brake

                mw.writerow(self._dict_measurements)

    def _create_log_files(self):
        """
        Just create the log files and add the necessary header for it.
        """

        if not self._experiment_exist():
            os.mkdir(self._path)

            with open(os.path.join(self._path, 'summary.csv'), 'w') as ofd:
                sw = csv.DictWriter(ofd, self._dict_summary.keys())
                sw.writeheader()
                if self._summary_fieldnames is None:
                    self._summary_fieldnames = sw.fieldnames

            with open(os.path.join(self._path, 'measurements.csv'), 'w') as rfd:
                mw = csv.DictWriter(rfd, self._dict_measurements.keys())
                mw.writeheader()
                if self._measurements_fieldnames is None:
                    self._measurements_fieldnames = mw.fieldnames

    def _continue_experiment(self, continue_experiment):
        """
        Get the line on the file for the experiment.
        If continue_experiment is false and experiment exist, generates a new file path

        """

        def get_non_existent_path(f_name_path):
            """
            Get the path to a filename which does not exist by incrementing path.
            """
            if not os.path.exists(f_name_path):
                return f_name_path
            filename, file_extension = os.path.splitext(f_name_path)
            i = 1
            new_f_name = "{}-{}{}".format(filename, i, file_extension)
            while os.path.exists(new_f_name):
                i += 1
                new_f_name = "{}-{}{}".format(filename, i, file_extension)
            return new_f_name

        # start the new path as the same one as before
        new_path = self._path
        summary_fieldnames = None
        measurements_fieldnames = None

        # if the experiment exist
        if self._experiment_exist():

            # If you want to continue just get the last position
            if continue_experiment:
                line_on_file = self._get_last_position()
                # Get the previously used fileorder
                with open(os.path.join(self._path, 'summary.csv'), 'r') as ofd:
                    summary_reader = csv.DictReader(ofd)
                    summary_fieldnames = summary_reader.fieldnames
                with open(os.path.join(self._path, 'measurements.csv'), 'r') as ofd:
                    measurements_reader = csv.DictReader(ofd)
                    measurements_fieldnames = measurements_reader.fieldnames

            else:
                # Get a new non_conflicting path name
                new_path = get_non_existent_path(new_path)
                line_on_file = 1

        else:
            line_on_file = 1
        return new_path, line_on_file, summary_fieldnames, measurements_fieldnames

    def save_images(self, sensor_data, episode_name, frame):
        """
        Save a image during the experiment
        """
        if self._save_images:
            for name, image in sensor_data.items():
                image.save_to_disk(self._image_filename_format.format(
                    episode_name, name, frame))

    def write_video(self, sensor_data):
        """
        Write images into a video during the experiment
        """
        if self._save_videos:
            for name, image in sensor_data.items():
                self.video_writer.write(image.data[..., ::-1])

    def write_processed_video(self, processed_image):
        if self._save_processed_videos:
            self.processed_video_writer.write(processed_image[..., ::-1])

    def policy_roll_out(self, measurements, control, episode_name, frame, directions, target):
        """
        Save the measurements and controls
        """
        if self._policy_roll_out:
            measurements_path = self._measurements_filename_format.format(episode_name, frame)
            measurements_folder = os.path.dirname(measurements_path)
            if not os.path.isdir(measurements_folder):
                os.makedirs(measurements_folder)
            with open(measurements_path, 'w') as fo:
                jsonObj = MessageToDict(measurements)
                jsonObj.update({'directions': directions})
                jsonObj.update({'target': MessageToDict(target)})
                fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

            control_path = self._control_filename_format.format(episode_name, frame)
            control_folder = os.path.dirname(control_path)
            if not os.path.isdir(control_folder):
                os.makedirs(control_folder)
            control_dict = {'steer': control.steer, 'throttle': control.throttle, 'brake': control.brake}
            np.save(control_path, control_dict)

    def get_pose_experiment_rep(self, number_poses_task, repetitions):
        """
        Based on the line in log file, return the current pose, experiment and repetition.
        If the line is zero, create new log files.

        """
        # Warning: assumes that all tasks have the same size
        line_on_file = self._get_last_position() - 1
        if line_on_file == 0:
            return 0, 0, 0
        else:
            return int(line_on_file/repetitions) % number_poses_task, \
                   line_on_file // (number_poses_task * repetitions), \
                   line_on_file % repetitions


    def _experiment_exist(self):

        return os.path.exists(self._path)

    def _get_last_position(self):
        """
        Get the last position on the summary experiment file
        With this you are able to continue from there

        Returns:
             int, position:
        """
        # Try to open, if the file is not found
        try:
            with open(os.path.join(self._path, 'summary.csv')) as f:
                return sum(1 for _ in f)
        except IOError:
            return 0
