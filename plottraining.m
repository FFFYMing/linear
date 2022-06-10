%% plot 10
test=bestEstep3998;
test=reshape(test,14,14);
test=(test-3.72)/(-1.06);
testp=test;
gd=mamama;  %this is the ground truth, should be imported
gd=reshape(gd,23,23);
gdp=reshape(gd(5:18,5:18),14,14);
errorp=abs(testp-gdp);

bottom = min(min(min(min(testp)), min(min(gdp))), min(min(errorp)));
top  = max(max(max(max(testp)), max(max(gdp))), max(max(errorp)));

subplot(1,3,1)
imagesc(testp)
colorbar
caxis manual
caxis([bottom top]);
title('Learning result')
subplot(1,3,2)
imagesc(gdp)
colorbar
caxis manual
caxis([bottom top]);
title('Original sample')
subplot(1,3,3)
imagesc(abs(testp-gdp))
colorbar
caxis manual
caxis([bottom top]);
title('Error')

%% plot 30
test=bestEstep1025;
test=reshape(test,34,34);
test=(test-3.72)/(-1.06);
testp=test;
gd=mamamama;  %this is the ground truth, should be imported
gd=reshape(gd,43,43);
gdp=reshape(gd(5:38,5:38),34,34);
errorp=abs(testp-gdp);

bottom = min(min(min(min(testp)), min(min(gdp))), min(min(errorp)));
top  = max(max(max(max(testp)), max(max(gdp))), max(max(errorp)));

subplot(1,3,1)
imagesc(testp)
colorbar
caxis manual
caxis([bottom top]);
title('Learning result')
subplot(1,3,2)
imagesc(gdp)
colorbar
caxis manual
caxis([bottom top]);
title('Original sample')
subplot(1,3,3)
imagesc(abs(testp-gdp))
colorbar
caxis manual
caxis([bottom top]);
title('Error')

%% plot 100
test=bestEstep124;
test=reshape(test,104,104);
test=(test-3.72)/(-1.06);
testp=test;
gd=mamamamama;  %this is the ground truth, should be imported
gd=reshape(gd,113,113);
gdp=reshape(gd(5:108,5:108),104,104);
errorp=abs(testp-gdp);

bottom = min(min(min(min(testp)), min(min(gdp))), min(min(errorp)));
top  = max(max(max(max(testp)), max(max(gdp))), max(max(errorp)));

subplot(1,3,1)
imagesc(testp)
colorbar
caxis manual
caxis([bottom top]);
title('Learning result')
subplot(1,3,2)
imagesc(gdp)
colorbar
caxis manual
caxis([bottom top]);
title('Original sample')
subplot(1,3,3)
imagesc(abs(testp-gdp))
colorbar
caxis manual
caxis([bottom top]);
title('Error')

%% plot 80
test=bestEstep1011;
test=reshape(test,84,84);
test=(test-3.72)/(-1.06);
testp=test;
gd=mamamamamama;  %this is the ground truth, should be imported
gd=reshape(gd,93,93);
gdp=reshape(gd(5:88,5:88),84,84);
errorp=abs(testp-gdp);

bottom = min(min(min(min(testp)), min(min(gdp))), min(min(errorp)));
top  = max(max(max(max(testp)), max(max(gdp))), max(max(errorp)));

subplot(1,3,1)
imagesc(testp)
colorbar
caxis manual
caxis([bottom top]);
title('Learning result')
subplot(1,3,2)
imagesc(gdp)
colorbar
caxis manual
caxis([bottom top]);
title('Original sample')
subplot(1,3,3)
imagesc(abs(testp-gdp))
colorbar
caxis manual
caxis([bottom top]);
title('Error')
