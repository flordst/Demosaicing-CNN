function ouput_hamilton
function process_images_in_folder(folder_path)
    
    tif_files = dir(fullfile(folder_path, '*.TIF'));
    num_files = length(tif_files);
  
    psnr_file = fopen('Q:\psnr_hamilton_values.txt', 'w');
    ssim_file = fopen('Q:\ssim_hamilton_values.txt', 'w');
    
    for k = 1:num_files
        file_name = fullfile(folder_path, tif_files(k).name);
        I = imread(file_name);
        
        Im = apply_demosaicing_hamilton(I);
        
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

function Im_demosaicked = apply_demosaicing_hamilton(I)
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
    figure, imshow(uint8(Im)), title('Mosaic image')

    %% Demosaicking - we will reconstruct a color image from the mosaic image
    Im = squeeze(sum(Im,3)); % 2D image

    %% Second order gradient - Hamilton & Adams's method
    % variation
    dH = abs(conv2(Im,[-1 0 1],'same')) + abs(conv2(Im,[-1 0 2 0 -1],'same')); 
    dV = abs(conv2(Im,[-1 0 1]','same')) + abs(conv2(Im,[-1 0 2 0 -1]','same')); 

    Fgh = [-1 2 2 2 -1]/4; 
    Fgd = [0 0 -1 0 0; 0 0 2 0 0; -1 2 4 2 -1; 0 0 2 0 0;0 0 -1 0 0]/8;
    Igh = conv2(Im,Fgh,'same');
    Igv = conv2(Im,Fgh','same');
    Igd = conv2(Im,Fgd,'same');

    Ig = zeros(size(Igd));
    Ig(dH>dV) = Igv(dH>dV);
    Ig(dH<dV) = Igh(dH<dV);
    Ig(dH==dV) = Igd(dH==dV);
    Ig = Ig.*(1-mG) + Im.*mG; % at G positions keep G values intact

    % R & B
    Ir = (Im - Ig) .* mR;
    Ib = (Im - Ig) .* mB;
    Frb = [1 2 1;2 4 2; 1 2 1]/4; % bilinear - constancy of color difference
    Ir = conv2(Ir,Frb,'same') + Ig;
    Ib = conv2(Ib,Frb,'same') + Ig;

    Y(:,:,1) = Ir;
    Y(:,:,2) = Ig;
    Y(:,:,3) = Ib;

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
process_images_in_folder('Q:\demosaicing\orignal') % duong dan thu muc chua anh groundtruth
end
