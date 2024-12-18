%% Task 2 - Group 2

% Define the main directory
main_dir = 'Test_set\';
    
% Define directories for images and ROIs
image_dir = fullfile(main_dir, 'images3\');
ROI_dir = fullfile(main_dir, 'ROIs3\');
matrix_dir = fullfile(main_dir, 'cells3\');
    
% Obtain the list of image files
image_files = dir(fullfile(image_dir, '*.tiff'));
    
% Count the total number of images
total_images = numel(image_files);
    
% Initialize variables 
global_Recall = 0;
global_Precision = 0;
global_F1 = 0;
best_F1 = 0;
best_image_results = struct('Recall', 0, 'Precision', 0, 'F1', 0, ...
        'name', '', 'TP', 0, 'FP', 0, 'FN', 0);
    
% Iterate over each image file
for i = 1:total_images

    image_name = image_files(i).name;
    base_name = strrep(image_name, '.tiff', '');
        
    % Load the corresponding data from the matrices
    GT_data = load(fullfile(matrix_dir, [base_name, '.mat']));
    GT_matrix = GT_data.cellLocationsData;
    cells_image = imread(fullfile(image_dir, image_name));
    ROI_mask = imread(fullfile(ROI_dir, [base_name, '_ROI.png']));
        
    % Process the image
    [image_trimmed, binary_mask, final_image] = processImage(cells_image, ROI_mask);
       
    % Detect cells
    [squares] = detectCells(binary_mask);
        
    % Calculate metrics for the current image
    [TP, FP, FN, jaccard_index] = calculateMetrics(GT_matrix, squares);

    % Calculate recall, precision, and F1
    [Recall, Precision, F1_measure] = calculateRPF(TP, FP, FN);
        
    % Log and display the image that was processed
    disp(image_name); 

    % Update global metrics
    global_Recall = global_Recall + Recall;
    global_Precision = global_Precision + Precision;
    global_F1 = global_F1 + F1_measure;
     
    % If it is a better result than the previous one, update
    if F1_measure > best_F1
        best_F1 = F1_measure;
        best_image_results = updateBestResults(best_image_results, ...
            final_image, image_trimmed, image_name, TP, FP, FN, jaccard_index, Recall, ...
            Precision, F1_measure, squares, GT_matrix);
    end

end
    
% Calculate global metrics
global_Recall = global_Recall / total_images;
global_Precision = global_Precision / total_images;
global_F1 = global_F1 / total_images;

% Display the best result
displayBestResults(best_image_results);

% Display global metrics - average of calculated metrics
displayAverageResults(global_Recall, global_Precision, global_F1);


%% Image processing

% Function to process the original image

function [image_trimmed, binary_mask, final_image] = processImage(cells_image, ROI_mask)
    
    ROI_mask = ROI_mask > 0;  % convert ROI mask to binary (non-zero values become 1)

    % Image pre-processing 
    cells_image_gray = rgb2gray(cells_image);  % convert to grayscale
    cells_image_smoothed = medfilt2(cells_image_gray, [3, 3]);  % smoothing using a median filter
    edge_image = edge(cells_image_smoothed, 'canny');  % edge detection using the Canny operator

    % Apply the ROI mask (isolate the region of interest in the image)
    image_trimmed = bsxfun(@times, edge_image, ROI_mask);

    % Cell segmentation using adaptive thresholding
    binary_mask = imbinarize(image_trimmed, 'adaptive', 'ForegroundPolarity', ...
        'dark', 'Sensitivity', 0.43); % thresholding sensitivity

    % Create the final image - all pixels outside the ROI are set to black
    final_image = cells_image;
    final_image(repmat(~ROI_mask, [1, 1, 3])) = 0;
    
end

% Function to detect cells

function [squares] = detectCells(binary_mask)

    % Detect cells within the previously calculated binary mask
    [centers, radii, ~] = imfindcircles(binary_mask, [20, 50], 'Sensitivity', 0.9);

    % Convert detected cell circles to squares
    squares = circlesToSquares(centers, radii);

end

% Helper function: convert detected circles into squares

function squares = circlesToSquares(centers, radii)
    squares = zeros(size(centers, 1), 4);

    for i = 1:size(centers, 1)
        x = centers(i, 1);
        y = centers(i, 2);
        r = radii(i);

        % Calculate the square coordinates based on the center and radius
        x1 = x-r;
        y1 = y-r;
        width = 2 * r;
        height = 2 * r;
        squares(i, :) = [x1, y1, width, height];
    end
end


%% Metric calculations

% Function to calculate TP, FP, FN + Jaccard index (for curiosity)

function [TP, FP, FN, jaccard_index] = calculateMetrics(GT_matrix, squares)

    TP = 0;
    FP = 0;
    FN = 0;

    % Threshold value for Jaccard index
    jaccard_threshold = 0.5;

    % Compare detected cells with ground truth
    for i = 1:size(GT_matrix, 1) % analyze all ground truth cells

        % Extract ground truth data
        GT_x = GT_matrix(i, 1); % x coordinate
        GT_y = GT_matrix(i, 2); % y coordinate
        GT_width = GT_matrix(i, 3); % width
        GT_height = GT_matrix(i, 4); % height
    
        coincident = false; % check if detected cells coincide with GT
    
        for j = 1:size(squares, 1) % analyze all detected cells

            % Extract data from detected cells
            cell_x = squares(j, 1); % x coordinate
            cell_y = squares(j, 2); % y coordinate
            cell_width = squares(j, 3); % width
            cell_height = squares(j, 4); % height

            % Calculate intersection area and union area
            intersection_area = calculateIntersectionArea(GT_x, GT_y, GT_width, GT_height, cell_x, cell_y, cell_width, cell_height);
            union_area = GT_width * GT_height + cell_width * cell_height - intersection_area;
        
            % Calculate Jaccard index
            jaccard_index = intersection_area / union_area;
        
            % Check if the Jaccard index exceeds the defined threshold
            if jaccard_index >= jaccard_threshold
                TP = TP + 1;
                coincident = true;
                break; % proceed to the next ground truth bounding box
            end
        end
   
        if ~coincident % if not coincident, count as a false negative
            FN = FN + 1;
        end
    end

    % Calculate false positives
    FP = size(squares, 1) - TP;

end 

% Function to calculate Recall, Precision, F1 measure

function [Recall, Precision, F1_measure] = calculateRPF(TP, FP, FN)
    Recall = TP / (TP + FN);
    Precision = TP / (TP + FP);
    F1_measure = 2 * (Precision * Recall) / (Precision + Recall);
end

% Helper function: calculate intersection between two squares

function intersection_area = calculateIntersectionArea(r1_x, r1_y, r1_width, r1_height, r2_x, r2_y, r2_width, r2_height)
    x_overlap = max(0, min(r1_x + r1_width, r2_x + r2_width) - max(r1_x, r2_x));
    y_overlap = max(0, min(r1_y + r1_height, r2_y + r2_height) - max(r1_y, r2_y));
    intersection_area = x_overlap * y_overlap;
end


%% Results

% Function to store the values of the best result image

function best_image_results = updateBestResults(best_image_results, ...
    final_image, image_trimmed, image_name, TP, FP, FN, jaccard_index, Recall, ...
    Precision, F1_measure, squares, GT_matrix)

    best_image_results.image = final_image;
    best_image_results.image_trimmed = image_trimmed;
    best_image_results.name = image_name;
    best_image_results.TP = TP;
    best_image_results.FP = FP;
    best_image_results.FN = FN;
    best_image_results.jaccard_index = jaccard_index;
    best_image_results.Recall = Recall;
    best_image_results.Precision = Precision;
    best_image_results.F1 = F1_measure;
    best_image_results.squares = squares;
    best_image_results.GT_matrix = GT_matrix;

end

% Function to display the image with the best result and its corresponding values

function displayBestResults(best_image_results)

    fprintf('\nBest Result:\n\n');
    fprintf('Number of counted cells: %d\n', ...
        size(best_image_results.squares, 1));
    fprintf('True Positives (TP): %d\n', best_image_results.TP);
    fprintf('False Positives (FP): %d\n', best_image_results.FP);
    fprintf('False Negatives (FN): %d\n', best_image_results.FN);
    fprintf('Jaccard Index: %.2f\n', best_image_results.jaccard_index);
    fprintf('Recall (R): %.2f\n', best_image_results.Recall);
    fprintf('Precision (P): %.2f\n', best_image_results.Precision);
    fprintf('F1-measure (F1): %.2f\n', best_image_results.F1);
    
    figure;
    imshow(best_image_results.image);
    title(['Best Image Processed: ', best_image_results.name]);
    hold on;

    % Plot detected cells - red squares
    for i = 1:size(best_image_results.squares, 1)
        square = best_image_results.squares(i,:);
        x = square(1);
        y = square(2);
        width = square(3);
        height = square(4);
        rectangle('Position', [x, y, width, height], 'EdgeColor', 'r');
    end

    % Plot GT cells - green rectangles
    for i = 1:size(best_image_results.GT_matrix, 1)
        rectangle('Position', ...
            [best_image_results.GT_matrix(i, 1), ...
            best_image_results.GT_matrix(i, 2), ...
            best_image_results.GT_matrix(i, 3), ...
            best_image_results.GT_matrix(i, 4)], 'EdgeColor', 'g');
    end

    hold off;
end

% Function to display final results - average of calculated metrics

function displayAverageResults(global_Recall, global_Precision, global_F1)

    fprintf('\nAverage Results:\n\n');
    fprintf('Average Recall (R): %.2f\n', global_Recall);
    fprintf('Average Precision (P): %.2f\n', global_Precision);
    fprintf('Average F1-measure (F1): %.2f\n\n', global_F1);
    
end
