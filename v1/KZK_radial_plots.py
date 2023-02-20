KZK_radial_plots(r,Ir,H,p5,p0,rho2,c2,R,a):

  """ 

  Produces the following plots:

  Heating rate vs. radius at focus
  Intensity vs. radius at focus
  Pressure amplitude of first (up to 5) harmonics vs. radius at focus

  """

  RR = a*R

  p5p5 = 1e-6*p0*p5

  IrIr = 1e-4*0.5*p0*p0*Ir/rho2/c2

  figure
  hold on
  plot(r,p5p5,'LineWidth',2)
  plot(-r,p5p5,'LineWidth',2)
  set(gca,'FontSize',12)
  ylim = get(gca,'YLim');
  axis([-RR,RR,ylim(1),ylim(2)])
  xlabel('r (cm)','FontSize',14)
  ylabel('p (MPa)','FontSize',14)
  title('Radial Pressure','FontSize',18)
  set(gca,'XMinorTick','on','YMinorTick','on')
  grid
  hold off

  figure
  hold on
  plot(r,IrIr,'LineWidth',2), 
  plot(-r,IrIr,'LineWidth',2)
  set(gca,'FontSize',12)
  ylim = get(gca,'YLim');
  axis([-RR,RR,ylim(1),ylim(2)])
  xlabel('r (cm)','FontSize',14)
  ylabel('I (W/cm^2)','FontSize',14)
  title('Radial Intensity','FontSize',18)
  set(gca,'XMinorTick','on','YMinorTick','on')
  grid
  hold off

  figure
  hold on
  plot(r,H,'r','LineWidth',2)
  plot(-r,H,'r','LineWidth',2)
  set(gca,'FontSize',12)
  ylim = get(gca,'YLim');
  axis([-RR,RR,ylim(1),ylim(2)])
  xlabel('r (cm)','FontSize',14)
  ylabel('H (W/cm^3)','FontSize',14)
  title('Radial Heating Rate','FontSize',18)
  set(gca,'XMinorTick','on','YMinorTick','on')
  grid
  hold off
  
  del IrIr, p5p5, RR
  