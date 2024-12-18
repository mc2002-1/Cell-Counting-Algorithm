%% Task 1 - Grupo 2

% Define the main directory
main_dir = 'Test_set\';

% Define the directories for images and ROIs
image_dir = fullfile(main_dir, 'images3/');
GT_dir = fullfile(main_dir, 'ROIs3/');
    
% Get the list of image files
image_files = dir(fullfile(image_dir, '*.tiff'));

% Count the total number of images
total_images = numel(image_files);

% Initialize the variables
total_jaccard_index = 0;
total_mean_euclidean_distance = 0;
total_max_euclidean_distance = 0;
best_image_name = '';
best_mean_euclidean_distance = Inf;
best_max_euclidean_distance = Inf;
best_jaccard_index = 0;
best_ROI_binary = [];
best_GT_mask = [];
best_original_image = [];

% Iterate over each image file
for i = 1:total_images
    
    image_name = image_files(i).name; % obter o nome da respetiva imagem
   
    original_image = imread(fullfile(image_dir, image_name)); % carregar a imagem
    original_image = im2double(original_image); % converter para double

    GT_name = strrep(image_name, '.tiff', '_ROI.png'); % obter o respetivo ROI
    GT_mask = imread(fullfile(GT_dir, GT_name)); % carregar o ROI
    
    % Image pre-processing
    gray_image = rgb2gray(original_image); % converter para cinzento
    enhanced_image = histeq(gray_image); % equalização do histograma - melhorar contraste
    
    % Detect the ROI
    ROI_binary = segmentation_method(enhanced_image);
    
    % Calculate the Jaccard index
    intersection = ROI_binary & GT_mask;
    union = ROI_binary | GT_mask;
    jaccard_index = sum(intersection(:)) / sum(union(:));
    
    % Calculate the Euclidean distances
    [mean_euclidean_distance, max_euclidean_distance] = calculate_euclidean_distances(ROI_binary, GT_mask);

    % Uncomment to show the ROI of each image:
    % figure;
    % subplot(1,2,1);
    % imshow(enhanced_image); 
    % title(['Image:' image_name]);
    % subplot(1,2,2);
    % imshow(ROI_binary);
    % title('Respective ROI');
    % pause(0.01);
   
    % Update global metrics
    total_jaccard_index = total_jaccard_index + jaccard_index;
    total_mean_euclidean_distance = total_mean_euclidean_distance + mean_euclidean_distance;
    total_max_euclidean_distance = total_max_euclidean_distance + max_euclidean_distance;
    
    % If this result is better than the previous, update
    if jaccard_index > best_jaccard_index && mean_euclidean_distance < best_mean_euclidean_distance && max_euclidean_distance < best_max_euclidean_distance
        best_jaccard_index = jaccard_index;
        best_mean_euclidean_distance = mean_euclidean_distance;
        best_max_euclidean_distance = max_euclidean_distance;
        best_image_name = image_name;
        best_ROI_binary = ROI_binary;
        best_GT_mask = GT_mask;
        best_original_image = original_image;
    end

end

% Calculate global metrics
avg_jaccard_index = total_jaccard_index / total_images;
avg_mean_euclidean_distance = total_mean_euclidean_distance / total_images;
avg_max_euclidean_distance = total_max_euclidean_distance / total_images;

% Display the best result
displayBestResults(best_image_name, best_jaccard_index, best_mean_euclidean_distance, best_max_euclidean_distance, best_ROI_binary, best_GT_mask, best_original_image);

% Display global metrics - average calculated metrics
displayAverageResults(avg_jaccard_index, avg_mean_euclidean_distance, avg_max_euclidean_distance);


%% Segmentation

% Segmentation function

function roi_binary = segmentation_method(enhanced_image)
    if enhanced_image(1200, 800) < 0.80 % look for the start of the 3 parallel lines
        for i = 1:2:150 
            if (enhanced_image(1200 - i, 800) > 0.80) && ...
               (enhanced_image(1200 - i - 24, 800) > 0.80) && ...
               (enhanced_image(1200 - i - 48, 800) > 0.80)
                y1 = 1200 - i - 24; % save a point from the middle line
                break;
            end
        end
        if enhanced_image(y1, 800) > 0.80 % look for the edges of the middle line
            m1 = y1; % variables for averaging
            m2 = y1;
            while enhanced_image(m1, 800) > 0.80
                m1 = m1 - 1;
            end
            while enhanced_image(m2, 800) > 0.80
                m2 = m2 + 1;
            end
            y1 = (m1 + m2) / 2;
        end
    else % if the initial point lands on a white line, use this else
        for i = 1:2:150
            if (enhanced_image(1200 - i, 830) > 0.80) && ...
               (enhanced_image(1200 - i - 24, 830) > 0.80) && ...
               (enhanced_image(1200 - i - 48, 830) > 0.80)
                y1 = 1200 - i - 24;
                break;
            end
        end
        if enhanced_image(y1, 830) > 0.80
            m1 = y1;
            m2 = y1;
            while enhanced_image(m1, 830) > 0.80
                m1 = m1 - 1;
            end
            while enhanced_image(m2, 830) > 0.80
                m2 = m2 + 1;
            end
            y1 = (m1 + m2) / 2;
        end
    end

    if enhanced_image(1, 800) < 0.80
        for i = 1:2:150
            if (enhanced_image(1 + i, 800) > 0.80) && ...
               (enhanced_image(1 + i + 24, 800) > 0.80) && ...
               (enhanced_image(1 + i + 48, 800) > 0.80)
                y2 = 1 + i + 24;
                break;
            end
        end
        if enhanced_image(y2, 800) > 0.80
            m1 = y2;
            m2 = y2;
            while enhanced_image(m1, 800) > 0.80
                m1 = m1 + 1;
            end
            while enhanced_image(m2, 800) > 0.80
                m2 = m2 - 1;
            end
            y2 = (m1 + m2) / 2;
        end
    else
        for i = 1:2:150
            if (enhanced_image(1 + i, 830) > 0.80) && ...
               (enhanced_image(1 + i + 24, 830) > 0.80) && ...
               (enhanced_image(1 + i + 48, 830) > 0.80)
                y2 = 1 + i + 24;
                break;
            end
        end
        if enhanced_image(y2, 830) > 0.80
            m1 = y2;
            m2 = y2;
            while enhanced_image(m1, 830) > 0.80
                m1 = m1 + 1;
            end
            while enhanced_image(m2, 830) > 0.80
                m2 = m2 - 1;
            end
            y2 = (m1 + m2) / 2;
        end
    end

    if enhanced_image(500, 1) < 0.80
        for i = 1:4:520
            if (enhanced_image(500, 1 + i) > 0.80) && ...
               (enhanced_image(500, 1 + i + 24) > 0.80) && ...
               (enhanced_image(500, 1 + i + 48) > 0.80)
                x1 = 1 + i + 24;
                break;
            end
        end
        if enhanced_image(500, x1) > 0.80
            m1 = x1;
            m2 = x1;
            while enhanced_image(500, m1) > 0.80
                m1 = m1 + 1;
            end
            while enhanced_image(500, m2) > 0.80
                m2 = m2 - 1;
            end
            x1 = (m1 + m2) / 2;
        end
    else
        for i = 1:4:520
            if (enhanced_image(530, 1 + i) > 0.80) && ...
               (enhanced_image(530, 1 + i + 24) > 0.80) && ...
               (enhanced_image(530, 1 + i + 48) > 0.80)
                x1 = 1 + i + 24;
                break;
            end
        end
        if enhanced_image(530, x1) > 0.80
            m1 = x1;
            m2 = x1;
            while enhanced_image(530, m1) > 0.80
                m1 = m1 + 1;
            end
            while enhanced_image(530, m2) > 0.80
                m2 = m2 - 1;
            end
            x1 = (m1 + m2) / 2;
        end
    end

    if enhanced_image(650, 1600) < 0.80
        for i = 1:5:700
            if (enhanced_image(650, 1600 - i) > 0.80) && ...
               (enhanced_image(650, 1600 - i - 25) > 0.80) && ...
               (enhanced_image(650, 1600 - i - 50) > 0.80)
                x2 = 1600 - i - 25;
                break;
            end
        end
        if enhanced_image(650, x2) > 0.80
            m1 = x2;
            m2 = x2;
            while enhanced_image(650, m1) > 0.80
                m1 = m1 - 1;
            end
            while enhanced_image(650, m2) > 0.80
                m2 = m2 + 1;
            end
            x2 = (m1 + m2) / 2;
        end
    else
        for i = 1:5:700
            if (enhanced_image(680, 1600 - i) > 0.80) && ...
               (enhanced_image(680, 1600 - i - 25) > 0.80) && ...
               (enhanced_image(680, 1600 - i - 50) > 0.80)
                x2 = 1600 - i - 25;
                break;
            end
        end
        if enhanced_image(680, x2) > 0.80
            m1 = x2;
            m2 = x2;
            while enhanced_image(680, m1) > 0.80
                m1 = m1 - 1;
            end
            while enhanced_image(680, m2) > 0.80
                m2 = m2 + 1;
            end
            x2 = (m1 + m2) / 2;
        end
    end

       % Confirmar que os índices são números inteiros
    y1 = round(y1);
    y2 = round(y2);
    x1 = round(x1);
    x2 = round(x2);
    

   % Instead of settling for the midpoint found above,
   % Retrieve a new point that attempts to reflect the angle of the lines;
   % To do this, we retrieved the upper boundary of the start of the line already inside
   % the square and the lower boundary of the end of the same line before exiting the square,
   % and the average between these was calculated


    m1=y1;     
    while enhanced_image(m1,x1+70) > 0.8
        m1=m1-1;
    end
    m2=y1;
    while enhanced_image(m2,x2-70) >0.8
        m2=m2+1;
    end
    y1=(m1+m2)/2;
    y1=round(y1);   %it's necessary to round because y1 will be                
    m1=y2;          %used as index
    while enhanced_image(m1,x1+70) > 0.8
        m1=m1-1;
    end
    m2=y2;
    while enhanced_image(m2,x2-70) > 0.8
        m2=m2+1;
    end
    y2=(m1+m2)/2;
    y2=round(y2);

    m1=x1;
    while enhanced_image(y2+70,m1) > 0.8
        m1=m1-1;
    end
    m2=x1;
    while enhanced_image(y1-70,m2) > 0.8
        m2=m2+1;
    end
    x1=(m1+m2)/2;
    x1=round(x1);

    m1=x2;
    while enhanced_image(y2+70,m1) > 0.8
        m1=m1-1;
    end
    m2=x2;
    while enhanced_image(y1-70,m2) >0.8
        m2=m2+1;
    end
    x2=(m1+m2)/2;
    x2=round(x2);

    % Create the mask
    roi_binary = zeros(size(enhanced_image));
    roi_binary(y2:y1, x1:x2) = 1;
 
end

%% Metric Calculation

% Function to calculate Euclidean distances

function [mean_euclidean_distance, max_euclidean_distance] = calculate_euclidean_distances(ROI_binary, GT_mask)
    
    % Find the first and last non-zero pixel and register the
    % coordinates for the binary mask of the ROI
    [ROI_x1, ROI_y1] = find(ROI_binary, 1, 'first');
    [ROI_x4, ROI_y4] = find(ROI_binary, 1, 'last');

    % Calculate the coordinates of the remaining vertices
    ROI_x2 = ROI_x1;
    ROI_y2 = ROI_y4;
    ROI_x3 = ROI_x4;
    ROI_y3 = ROI_y1;

    % Construct the vertices for the ROI
    ROI_vertices = [ROI_x1, ROI_y1; ROI_x2, ROI_y2; ROI_x3, ROI_y3; ROI_x4, ROI_y4];

    % Find the first and last non-zero pixel and register the
    % coordinates for the ground truth mask
    [GT_x1, GT_y1] = find(GT_mask, 1, 'first');
    [GT_x4, GT_y4] = find(GT_mask, 1, 'last');

    % Calculate the coordinates of the remaining vertices
    GT_x2 = GT_x1;
    GT_y2 = GT_y4;
    GT_x3 = GT_x4;
    GT_y3 = GT_y1;

    % Construct the vertices for the ground truth
    GT_vertices = [GT_x1, GT_y1; GT_x2, GT_y2; GT_x3, GT_y3; GT_x4, GT_y4];

    % Calculate the Euclidean distance between the vertices of the ROI and GT
    distances = sqrt(sum((ROI_vertices - GT_vertices).^2, 2));
    
    % Calculate the mean of the distances and the maximum distance 
    mean_euclidean_distance = mean(distances);
    max_euclidean_distance = max(distances);

end


%% Results

% Function to display the best results obtained along with the respective image

function displayBestResults(best_image_name, best_jaccard_index, best_mean_euclidean_distance, best_max_euclidean_distance, best_ROI_binary, best_GT_mask, best_original_image)
    
    fprintf('\nBest Results:\n\n');
    fprintf('Best Image Processed: %s\n', best_image_name); 
    fprintf('Best Jaccard Index: %.2f\n', best_jaccard_index);
    fprintf('Best Mean Euclidean Distance: %.2f\n', best_mean_euclidean_distance);
    fprintf('Best Max Euclidean Distance: %.2f\n', best_max_euclidean_distance);

    % Extract the region of interest from the image - place the best calculated ROI in the respective image
    image_trimmed = bsxfun(@times, best_original_image, double(best_ROI_binary));

    figure;
    subplot(2, 2, 1); imshow(best_original_image); title('Original Image');
    subplot(2, 2, 3); imshow(image_trimmed); title('Original Image with Detected ROI');
    subplot(2, 2, 2); imshow(best_ROI_binary); title('Detected ROI');
    subplot(2, 2, 4); imshow(best_GT_mask); title('Ground Truth ROI');

end

% Function to display the final results - averages of the calculated metrics

function displayAverageResults(avg_jaccard_index, avg_mean_euclidean_distance, avg_max_euclidean_distance)

    fprintf('\nAverage Results:\n\n');
    fprintf('Average Jaccard Index: %.2f\n', avg_jaccard_index);
    fprintf('Average Mean Euclidean Distance: %.2f\n', avg_mean_euclidean_distance);
    fprintf('Average Max Euclidean Distance: %.2f\n', avg_max_euclidean_distance); % Is this MAX or MIN or both?

end
