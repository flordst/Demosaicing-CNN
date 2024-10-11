function output_laroche
function process_images_in_folder(folder_path)
    
    tif_files = dir(fullfile(folder_path, '*.TIF'));
    num_files = length(tif_files);
    
    psnr_file = fopen('Q:\psnr_values_laroche.txt', 'w');
    ssim_file = fopen('Q:\ssim_values_laroche.txt', 'w');
    
    for k = 1:num_files
        
        file_name = fullfile(folder_path, tif_files(k).name);
        I = imread(file_name);
              
        Im_demosaiced = apply_demosaicing_laroche(I);       
       
        psnr_value = PSNR_rgb_average(I, Im_demosaiced);
        ssim_value = f_SSIM(I, Im_demosaiced);
               
        fprintf(psnr_file, 'File: %s, PSNR: %.4f dB\n', tif_files(k).name, psnr_value);
        fprintf(ssim_file, 'File: %s, SSIM: %.4f\n', tif_files(k).name, ssim_value);
    end
    
   
    fclose(psnr_file);
    fclose(ssim_file);
    
end

function Im_demosaiced = apply_demosaicing_laroche(I)
    %% CFA sensor
    % filter array
    mR = zeros(size(I(:,:,1))); mR(1:2:end,1:2:end) = 1;
    mB = zeros(size(mR)); mB(2:2:end,2:2:end) = 1;
    mG = 1 - mR - mB;
    
    % mosaic image
    Im = double(I);
    Im(:,:,1) = Im(:,:,1) .* mR;
    Im(:,:,2) = Im(:,:,2) .* mG;
    Im(:,:,3) = Im(:,:,3) .* mB;
    
    % In reality we only have a mosaic image Im
    Im = squeeze(sum(Im, 3)); % 2D image
    
    %% Edge directed interpolation - Laroche's method
    % variation
    dH = conv2(Im, [-1 0 2 0 -1], 'same'); dH = abs(dH);
    dV = conv2(Im, [-1 0 2 0 -1]', 'same'); dV = abs(dV);
    Ig = Im .* mG;
    Fgh = [1 2 1] / 2; % at G position, the G value will not change
    Fgd = [0 1 0; 1 4 1; 0 1 0] / 4;
    Igh = conv2(Ig, Fgh, 'same');
    Igv = conv2(Ig, Fgh', 'same');
    Igd = conv2(Ig, Fgd, 'same');
    
    Ig(dH > dV) = Igv(dH > dV);
    Ig(dH < dV) = Igh(dH < dV);
    Ig(dH == dV) = Igd(dH == dV);
    
    % R & B interpolation
    Ir = (Im - Ig) .* mR;
    Ib = (Im - Ig) .* mB;
    Frb = [1 2 1; 2 4 2; 1 2 1] / 4; % bilinear - constancy of color difference
    Ir = conv2(Ir, Frb, 'same') + Ig;
    Ib = conv2(Ib, Frb, 'same') + Ig;
    
    Im_demosaiced(:,:,1) = Ir;
    Im_demosaiced(:,:,2) = Ig;
    Im_demosaiced(:,:,3) = Ib;
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
    max_pixel = 255.0;
    psnr_channel_value = 20 * log10(max_pixel / sqrt(mse));
end

function s = f_SSIM(img1, img2)
    img1_gray = rgb2gray(uint8(img1));
    img2_gray = rgb2gray(uint8(img2));
    s = ssim(img1_gray, img2_gray);
end


process_images_in_folder('Q:\demosaicing\orignal') % duong dan thu muc chua anh groundtruth 
end