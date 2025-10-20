load("kspace_Subject0023_ff.mat", "kspace");
USkspace = kspace; % 80 x 82 x 22 x 100
load("kspace_Subject0026_ff.mat","kspace");
FSkspace = kspace; % 312 x 410 x 22

% FS coil 1 kspace
figure;
subplot(1,2,1);
imagesc(log(abs(FSkspace(:,:,1)))); 
colormap('gray');
title(sprintf('FS kspace Subject0026 aa coil 1'));

% US coil 1 frame 1 kspace
subplot(1,2,2);
imagesc(fftshift(log(abs(USkspace(:,:,1))))); 
colormap('gray');
title(sprintf('US kspace Subject0023 ff coil 1 frame 1'));
%% FS image
% initialise coil images
FSx = 312;
FSy = 410;
coils = 22;

FS_img = zeros(FSx,FSy,coils);

for coil = 1:coils % looping over each coil    
    % extract coil image for the frame
    % image_coil = ifftshift(ifft2(fftshift(FSkspace(:,:,coil))));
    image_coil = ifft2(FSkspace(:,:,coil));
    FS_img(:,:,coil)=squeeze(FS_img(:,:,coil))+image_coil(:,:);
end

% combine across coils via sum of squares
FS_img_ss = sqrt(sum(abs(FS_img).^2,3)); % coil is the 3rd dimension
figure;
imagesc(fftshift(abs(FS_img_ss),1)); % This is the fix
colormap('gray'); axis image off;
title(sprintf('sum of squares'));

%% US image
% initialise coil images
USx = 80;
USy = 82;
coils = 22;

US_img = zeros(USx,USy,coils);

for coil = 1:coils % looping over each coil    
    % extract coil image for the frame
    image_coil = ifftshift(ifft2(USkspace(:,:,coil,1)));
    US_img(:,:,coil)=squeeze(US_img(:,:,coil))+image_coil(:,:);
end

% combine across coils via sum of squares
US_img_ss = sqrt(sum(abs(US_img).^2,3)); % coil is the 3rd dimension
figure;
imagesc(fftshift(abs(US_img_ss),2));
colormap('gray'); axis image off;
title(sprintf('sum of squares'));

