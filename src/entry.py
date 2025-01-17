"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
import platform
from pprint import pprint
from time import time

import cv2
from matplotlib import pyplot as plt
import pandas as pd
from rich.table import Table

from src import constants
from src.defaults import CONFIG_DEFAULTS
from src.evaluation import EvaluationConfig, evaluate_concatenated_response
from src.logger import console, logger
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.parsing import get_concatenated_response, open_config_with_defaults

# Load processors
STATS = Stats()
def hi():
    #Wrong...we want to initialize a template and configure manually. Could keep a base template and then make changes dynamically...let's see...
    template = Template("template_path", tuning_config="hi")
    #TODO Generate custom labels and field_blocks
    # well...possible however currently, field_block gen is is happening at a preprocessing level...let's take a closer look... a processor is expected to return an image...
    # we need to return the x coordinates divided by 3 and draw individual bounding boxes...so this preprocessing step is nice so that we can ensure rectangle detection is working
    # however, it doesn't really help us...so how do we connect these two pieces of code in an effective way?
    # well perhaps there should be a post preprocessing step that's before the processing step...rendundant but need the granularity
    template.custom_labels = ""
    template.field_blocks = ""
    
def entry_point(input_dir, args):
    if not os.path.exists(input_dir):
        raise Exception(f"Given input directory does not exist: '{input_dir}'")
    curr_dir = input_dir
    return process_dir(input_dir, curr_dir, args)


def process_dir(
    root_dir,
    curr_dir,
    args,
    template=None,
    tuning_config=CONFIG_DEFAULTS,
    evaluation_config=None,
):
    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    if os.path.exists(local_config_path):
        tuning_config = open_config_with_defaults(local_config_path)
    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    local_template_exists = os.path.exists(local_template_path)
    
    if local_template_exists:
        template = Template(
            local_template_path,
            tuning_config,
        )
    # Look for subdirectories for processing
    subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]

    output_dir = Path(args["output_dir"], curr_dir.relative_to(root_dir))
    paths = Paths(output_dir)

    # look for images in current dir to process
    if platform.system() == "Windows":
        exts = ("*.png", "*.jpg", "*.jpeg")
    else:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    omr_files = sorted([f for ext in exts for f in curr_dir.glob(ext)])

    # Exclude images (take union over all pre_processors)
    excluded_files = []
    if template:
        for pp in template.pre_processors:
            excluded_files.extend(Path(p) for p in pp.exclude_files())

    omr_files = [f for f in omr_files if f not in excluded_files]

    if omr_files:
        if not template:
            logger.error(
                f"Found images, but no template in the directory tree \
                of '{curr_dir}'. \nPlace {constants.TEMPLATE_FILENAME} in the \
                appropriate directory."
            )
            raise Exception(
                f"No template file found in the directory tree of {curr_dir}"
            )
        logger.info("")
        table = Table(
            title="Current Configurations", show_header=False, show_lines=False
        )
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("Directory Path", f"{curr_dir}")
        table.add_row("Count of Images", f"{len(omr_files)}")
        table.add_row(
            "Markers Detection",
            "ON" if "CropOnMarkers" in template.pre_processors else "OFF",
        )
        table.add_row("Auto Alignment", f"{tuning_config.alignment_params.auto_align}")
        table.add_row("Detected Template Path", f"{template}")
        table.add_row(
            "Detected pre-processors",
            f"{[pp.__class__.__name__ for pp in template.pre_processors]}",
        )
        console.print(table, justify="center")

        setup_dirs_for_paths(paths)
        outputs_namespace = setup_outputs_for_template(paths, template)
        if args["setLayout"]:
            show_template_layouts(omr_files, template, tuning_config)
        else:
            local_evaluation_path = curr_dir.joinpath(constants.EVALUATION_FILENAME)
            if os.path.exists(local_evaluation_path):
                if not local_template_exists:
                    logger.warning(
                        f"Found an evaluation file without a parent template file: {local_evaluation_path}"
                    )
                evaluation_config = EvaluationConfig(
                    local_evaluation_path, template, curr_dir
                )

                excluded_files.extend(
                    Path(exclude_file)
                    for exclude_file in evaluation_config.get_exclude_files()
                )

            omr_files = [f for f in omr_files if f not in excluded_files]
            process_files(
                omr_files,
                template,
                tuning_config,
                evaluation_config,
                outputs_namespace,
            )
            
    elif not subdirs:
        # Each subdirectory should have images or should be non-leaf
        logger.info(
            f"No valid images or sub-folders found in {curr_dir}.\
            Empty directories not allowed."
        )

    # recursively process sub-folders
    for d in subdirs:
        process_dir(
            root_dir,
            d,
            args,
            template,
            tuning_config,
            evaluation_config,
        )

def show_template_layouts(omr_files, template, tuning_config):
    for file_path in omr_files:
        file_name = file_path.name
        file_path = str(file_path)
        in_omr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        #preprocessing is applied in the template.
        print("Before generate_template:", template.bubble_dimensions)
        message = len(template.field_blocks_object)
        logger.info("🚀 ~ file: entry.py:183 ~ message:", message)
        #TODO image_instance_ops.draw_rectangle_layout
        template = template.image_instance_ops.generate_template(in_omr, template)
        # currenty 180...which is 4 cols of 15. 3 rows each. So we can validate to be sure there are 180 total blocks.
        pprint(template.field_blocks_object, indent=4)
        template.finish_setup()
        print("After generate_template:", template.bubble_dimensions)
        template_layout = template.image_instance_ops.draw_template_layout(
            in_omr, template, shifted=False, border=2
        )
        print("After draw_template_layout:", template.bubble_dimensions)
        
        InteractionUtils.show(
            f"Template Layout: {file_name}", template_layout, 1, 1, config=tuning_config
        )
        
def show_rectangle_layouts(omr_files, template, tuning_config):
    for file_path in omr_files:
        file_name = file_path.name
        file_path = str(file_path)
        in_omr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        in_omr = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection to the grayscale image
        edged = cv2.Canny(in_omr, 50, 200)

        # Find contours in the edge map
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a list to store the coordinates of the rectangles
        rect_coords = []

        # Iterate through the contours and filter for rectangles
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                rect_coords.append((x, y, w, h))

        # Sort the rectangles by their y-coordinate first and then by their x-coordinate
        rect_coords = sorted(rect_coords, key=lambda x: (x[1], x[0]))

        # Draw the rectangles on the image in the sorted order
        for i, (x, y, w, h) in enumerate(rect_coords):
            cv2.rectangle(in_omr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print("Rectangle %d: (x=%d, y=%d, w=%d, h=%d)" % (i+1, x, y, w, h))

        # Show the detected rectangles on the original image
        plt.imshow(cv2.cvtColor(in_omr, cv2.COLOR_BGR2RGB))
        plt.show()
        
        
        
        
        template_layout = template.image_instance_ops.draw_template_layout(
            in_omr, template, shifted=False, border=2
        )
        InteractionUtils.show(
            f"Template Layout: {file_name}", template_layout, 1, 1, config=tuning_config
        )


def process_files(
    omr_files,
    template,
    tuning_config,
    evaluation_config,
    outputs_namespace,
):
    start_time = int(time())
    files_counter = 0
    STATS.files_not_moved = 0

    for file_path in omr_files:
        files_counter += 1
        file_name = file_path.name

        in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        logger.info("")
        logger.info(
            f"({files_counter}) Opening image: \t'{file_path}'\tResolution: {in_omr.shape}"
        )
        # reset list of images to be saved
        template.image_instance_ops.reset_all_save_img()
        # append initial image
        template.image_instance_ops.append_save_img(1, in_omr)
        # apply preprocessing - in my case simply crop
        in_omr = template.image_instance_ops.apply_preprocessors(
            file_path, in_omr, template
        )
        ##TODO Generate Custom Labels, also handle generate_template_flag here as well
        if template.autoGenerateTemplate:
            template = template.image_instance_ops.generate_template(in_omr, template)
        template.finish_setup()

        if in_omr is None:
            # Error OMR case
            new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
            outputs_namespace.OUTPUT_SET.append(
                [file_name] + outputs_namespace.empty_resp
            )
            if check_and_move(
                constants.ERROR_CODES.NO_MARKER_ERR, file_path, new_file_path
            ):
                err_line = [
                    file_name,
                    file_path,
                    new_file_path,
                    "NA",
                ] + outputs_namespace.empty_resp
                pd.DataFrame(err_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["Errors"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            continue

        # uniquify
        file_id = str(file_name)
        save_dir = outputs_namespace.paths.save_marked_dir
        (
            response_dict,
            final_marked,
            multi_marked,
            _,
        ) = template.image_instance_ops.read_omr_response(
            template, image=in_omr, name=file_id, save_dir=save_dir
        )

        # TODO: move inner try catch here
        # concatenate roll nos, set unmarked responses, etc
        omr_response = get_concatenated_response(response_dict, template)

        if (
            evaluation_config is None
            or not evaluation_config.get_should_explain_scoring()
        ):
            logger.info(f"Read Response: \n{omr_response}")

        score = 0
        if evaluation_config is not None:
            score = evaluate_concatenated_response(omr_response, evaluation_config)
            logger.info(
                f"(/{files_counter}) Graded with score: {round(score, 2)}\t for file: '{file_id}'"
            )
        else:
            logger.info(f"(/{files_counter}) Processed file: '{file_id}'")

        if tuning_config.outputs.show_image_level >= 2:
            InteractionUtils.show(
                f"Final Marked Bubbles : '{file_id}'",
                ImageUtils.resize_util_h(
                    final_marked, int(tuning_config.dimensions.display_height * 1.3)
                ),
                1,
                1,
                config=tuning_config,
            )

        resp_array = []
        for k in template.output_columns:
            resp_array.append(omr_response[k])

        outputs_namespace.OUTPUT_SET.append([file_name] + resp_array)

        if multi_marked == 0:
            STATS.files_not_moved += 1
            new_file_path = save_dir.joinpath(file_id)
            # Enter into Results sheet-
            results_line = [file_name, file_path, new_file_path, score] + resp_array
            # Write/Append to results_line file(opened in append mode)
            pd.DataFrame(results_line, dtype=str).T.to_csv(
                outputs_namespace.files_obj["Results"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
        else:
            # multi_marked file
            logger.info(f"[{files_counter}] Found multi-marked file: '{file_id}'")
            new_file_path = outputs_namespace.paths.multi_marked_dir.joinpath(file_name)
            if check_and_move(
                constants.ERROR_CODES.MULTI_BUBBLE_WARN, file_path, new_file_path
            ):
                mm_line = [file_name, file_path, new_file_path, "NA"] + resp_array
                pd.DataFrame(mm_line, dtype=str).T.to_csv(
                    outputs_namespace.files_obj["MultiMarked"],
                    mode="a",
                    quoting=QUOTE_NONNUMERIC,
                    header=False,
                    index=False,
                )
            # else:
            #     TODO:  Add appropriate record handling here
            #     pass

    print_stats(start_time, files_counter, tuning_config)


def check_and_move(error_code, file_path, filepath2):
    # TODO: fix file movement into error/multimarked/invalid etc again
    STATS.files_not_moved += 1
    return True


def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
    log = logger.info
    log("")
    log(f"{'Total file(s) moved':<27}: {STATS.files_moved}")
    log(f"{'Total file(s) not moved':<27}: {STATS.files_not_moved}")
    log("--------------------------------")
    log(
        f"{'Total file(s) processed':<27}: {files_counter} ({'Sum Tallied!' if files_counter == (STATS.files_moved + STATS.files_not_moved) else 'Not Tallying!'})"
    )

    if tuning_config.outputs.show_image_level <= 0:
        log(
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking/60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate':<27}:\t ~ {round(time_checking/files_counter,2)} seconds/OMR"
        )
        log(
            f"{'OMR Processing Speed':<27}:\t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute"
        )
    else:
        log(f"\n{'Total script time':<27}: {time_checking} seconds")

    if tuning_config.outputs.show_image_level <= 1:
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )
