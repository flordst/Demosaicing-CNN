function demosaicking_comparison
    I = imread('C:\Program Files\Polyspace\R2021a\bin\Image process\1_original.TIF');  % duong dan chua anh grountruth
    output1 = imread('C:\Program Files\Polyspace\R2021a\bin\Image process\1_result.TIF');  % duong dan chua anh output model REDNet20
    output2 = imread('C:\Program Files\Polyspace\R2021a\bin\Image process\1_result1.TIF'); % duong dan chua anh output model SRCNN
    
    % 3 thuat toan truyen thong
    Y_bilinear = demosaicing_bilinear(I);
    Y_second_order_grad = demosaicing_hamilton(I);
    Y_edge_directed = demosaicing_Laroche(I);

    % Tinh PSNR and SSIM 
    psnr_bilinear = PSNR_rgb_average(I, Y_bilinear);
    psnr_second_order_grad = PSNR_rgb_average(I, Y_second_order_grad);
    psnr_edge_directed = PSNR_rgb_average(I, Y_edge_directed);
    psnr_output1 = PSNR_rgb_average(I, output1);
    psnr_output2 = PSNR_rgb_average(I, output2);

    ssim_bilinear = ssim(uint8(Y_bilinear), I);
    ssim_second_order_grad = ssim(uint8(Y_second_order_grad), I);
    ssim_edge_directed = ssim(uint8(Y_edge_directed), I);
    ssim_output1 = ssim(output1, I);
    ssim_output2 = ssim(output2, I);
    figure;
 
    subplot(2,3,1), imshow(I), title('Groundtruth Image');
    
    subplot(2, 3, 2), imshow(uint8(output1)), title('REDNet20');
    xlabel(sprintf('PSNR: %.4f dB\nSSIM: %.4f', psnr_output1, ssim_output1));
    
    subplot(2, 3, 3), imshow(uint8(output2)), title('SRCNN');
    xlabel(sprintf('PSNR: %.4f dB\nSSIM: %.4f', psnr_output2, ssim_output2));

    subplot(2,3,4), imshow(uint8(Y_bilinear)), title('Bilinear Interpolation');
    xlabel(sprintf('PSNR: %.4f dB\nSSIM: %.4f', psnr_bilinear,ssim_bilinear));
    
    subplot(2,3,5), imshow(uint8(Y_second_order_grad)), title('Hamilton-Adam');
    xlabel(sprintf('PSNR: %.4f dB\nSSIM: %.4f', psnr_second_order_grad,ssim_second_order_grad));
    
    subplot(2,3,6), imshow(uint8(Y_edge_directed)), title('Laroche');
    xlabel(sprintf('PSNR: %.4f dB\nSSIM: %.4f', psnr_edge_directed,ssim_edge_directed));
    
    
end

function Y = demosaicing_bilinear(I)
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

    %% Bilinear interpolation
    % filtering
    Frb = [1 2 1; 2 4 2; 1 2 1]/4;
    Fg = [0 1 0; 1 4 1; 0 1 0]/4;

    Im_R = Im(:,:,1);
    Im_G = Im(:,:,2);
    Im_B = Im(:,:,3);
    Y(:,:,1) = conv2(Im_R, Frb, 'same');
    Y(:,:,2) = conv2(Im_G, Fg, 'same');
    Y(:,:,3) = conv2(Im_B, Frb, 'same');
end

function Y = demosaicing_hamilton(I)
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

function Y = demosaicing_Laroche(I)
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
    Im = squeeze(sum(Im,3)); % 2D image

    %% Edge directed interpolation - Laroche's method
    % variation
    dH = conv2(Im,[-1 0 2 0 -1],'same'); dH = abs(dH);
    dV = conv2(Im,[-1 0 2 0 -1]','same'); dV = abs(dV);
    Ig = Im.*mG;
    Fgh = [1 2 1]/2; % at G position, the G value will not change
    Fgd = [0 1 0;1 4 1;0 1 0]/4;
    Igh = conv2(Ig,Fgh,'same');
    Igv = conv2(Ig,Fgh','same');
    Igd = conv2(Ig,Fgd','same');

    Ig(dH>dV) = Igv(dH>dV);
    Ig(dH<dV) = Igh(dH<dV);
    Ig(dH==dV) = Igd(dH==dV);

    % R & B interpolation
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
    
    % Calculate PSNR for each channel
    psnr_R = PSNR_channel(original(:,:,1), compressed(:,:,1));
    psnr_G = PSNR_channel(original(:,:,2), compressed(:,:,2));
    psnr_B = PSNR_channel(original(:,:,3), compressed(:,:,3));
    
    % Calculate average PSNR
    psnr_value = mean([psnr_R, psnr_G, psnr_B]);
end

function psnr_channel_value = PSNR_channel(channel1, channel2)
    % Mean Squared Error (MSE)
    mse = mean((channel1 - channel2).^2, 'all');
    
    if mse == 0
        psnr_channel_value = 100;
        return;
    end
    
    % Calculate PSNR for the channel
    max_pixel = 255.0;
    psnr_channel_value = 20 * log10(max_pixel / sqrt(mse));
end

