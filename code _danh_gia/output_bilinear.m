function ouput_bilinear
function process_images_in_folder(folder_path)
    
    tif_files = dir(fullfile(folder_path, '*.TIF'));
    num_files = length(tif_files);
    
  
    psnr_file = fopen('Q:\psnr__bilinear_values.txt', 'w');
    ssim_file = fopen('Q:\ssim__bilinear_values.txt', 'w');
    
    for k = 1:num_files
        % Đọc ảnh
        file_name = fullfile(folder_path, tif_files(k).name);
        I = imread(file_name);
        
        % Thực hiện demosaicing bilinear
        Im = apply_demosaicing_bilinear(I);
        
        % Tính toán PSNR và SSIM
        psnr_value = PSNR_rgb_average(I, Im);
        ssim_value = f_SSIM(I, Im);
        
        % Ghi giá trị vào file
        fprintf(psnr_file, 'File: %s, PSNR: %.4f dB\n', tif_files(k).name, psnr_value);
        fprintf(ssim_file, 'File: %s, SSIM: %.4f\n', tif_files(k).name, ssim_value);
    end
    
   
    fclose(psnr_file);
    fclose(ssim_file);
    
    fprintf('Đã lưu giá trị PSNR và SSIM vào file psnr_values.txt và ssim_values.txt\n');
end

function Im_demosaicked = apply_demosaicing_bilinear(I)
    % CFA sensor
    mR = zeros(size(I(:,:,1))); mR(1:2:end,1:2:end) = 1;
    mB = zeros(size(mR)); mB(2:2:end,2:2:end) = 1;
    mG = 1 - mR - mB;
    
    % Mosaic image
    Im = double(I);
    Im(:,:,1) = Im(:,:,1) .* mR;
    Im(:,:,2) = Im(:,:,2) .* mG;
    Im(:,:,3) = Im(:,:,3) .* mB;

    % Bilinear interpolation
    Frb = [1 2 1; 2 4 2; 1 2 1] / 4;
    Fg = [0 1 0; 1 4 1; 0 1 0] / 4;
    
    Im_R = Im(:,:,1);
    Im_G = Im(:,:,2);
    Im_B = Im(:,:,3);
    
    Im_demosaicked(:,:,1) = conv2(Im_R, Frb, 'same');
    Im_demosaicked(:,:,2) = conv2(Im_G, Fg, 'same');
    Im_demosaicked(:,:,3) = conv2(Im_B, Frb, 'same');
end

function psnr_value = PSNR_rgb_average(original, compressed)
 
    original = double(original);
    compressed = double(compressed);
    
    % Tính PSNR cho từng kênh
    psnr_R = PSNR_channel(original(:,:,1), compressed(:,:,1));
    psnr_G = PSNR_channel(original(:,:,2), compressed(:,:,2));
    psnr_B = PSNR_channel(original(:,:,3), compressed(:,:,3));
    
    % Tính giá trị PSNR trung bình
    psnr_value = mean([psnr_R, psnr_G, psnr_B]);
end

function psnr_channel_value = PSNR_channel(channel1, channel2)
   
    mse = mean((channel1 - channel2).^2, 'all');   
    if mse == 0
        psnr_channel_value = 100;
        return;
    end
    
    % Tính giá trị PSNR cho kênh
    max_pixel = 255.0;
    psnr_channel_value = 20 * log10(max_pixel / sqrt(mse));
end

function s = f_SSIM(img1, img2)
    img1_gray = rgb2gray(uint8(img1));
    img2_gray = rgb2gray(uint8(img2));
    s = ssim(img1_gray, img2_gray);
end
process_images_in_folder('Q:\demosaicing\orignal') % duong dan thu muc chua anh grountruth
end
