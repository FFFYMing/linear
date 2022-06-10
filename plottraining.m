test=bestEstep45;
range=5:38;
test=reshape(test,34,34);
test=(test-3.72)/(-1.06);
testp=test;
gd=reshape(mamamama,43,43);  %this is the ground truth, should be generated in trans_to_100x100.m
gdp=gd(range,range);
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

%%
E_tanh=reshape(bestEstep390,14,14);
E_E=reshape(bestE,23,23);
E_E=E_E(5:18,5:18);
E_gd=reshape(mamama,23,23);
E_gd=E_gd(5:18,5:18);
E_gd=-1.06*E_gd+3.76;

loss_tanh=norm(E_tanh-E_gd)/norm(E_gd);
loss_E=norm(E_E-E_gd)/norm(E_gd);

bottom = min(min(min(min(abs(E_tanh-E_gd))), min(min(abs(E_E-E_gd)))));
top  = max(max(max(max(abs(E_tanh-E_gd))), max(max(abs(E_tanh-E_gd)))));

subplot(1,2,1)
imagesc(abs(E_tanh-E_gd))
colorbar
caxis manual
caxis([bottom top]);
title(['using tanh, loss=',num2str(loss_tanh)])
subplot(1,2,2)
imagesc(abs(E_E-E_gd))
colorbar
caxis manual
caxis([bottom top]);
title(['not using tanh, loss=',num2str(loss_E)])

%%
test=bestEstep3998;
test=reshape(test,14,14);
test=(test-3.72)/(-1.06);
testp=test;
gd=mamama;  %this is the ground truth, should be generated in trans_to_100x100.m
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
gd=mamamama;  %this is the ground truth, should be generated in trans_to_100x100.m
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
gd=mamamamama;  %this is the ground truth, should be generated in trans_to_100x100.m
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
gd=mamamamamama;  %this is the ground truth, should be generated in trans_to_100x100.m
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