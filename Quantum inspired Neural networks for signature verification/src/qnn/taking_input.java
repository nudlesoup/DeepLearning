@Author:AmeyaDhamanaskar
package qnn;

import java.util.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
public class taking_input extends functions
{
	static int feature = 45;
	static int persons = 37;
	static int realsign = 8;
	static double [][][] allsigns = new double [persons][realsign][feature];
	static double [][] person_under_consideration = new double [realsign][feature];
	static // these are being used in the updation function also so declared them here
     int bitworand1[] = new int [feature];
	static int bitworand2[] = new int [feature];
	static int bitworand3[] = new int [feature];
	
	

	static int tempbitworand1[] = new int [feature];
	static int tempbitworand2[] = new int [feature];
	static int tempbitworand3[] = new int [feature];
	// max ad min funcitons
	 public static double find_max ( interior_node a)
	 {
		
		
		double max =-10000.00;
		double min = 10000.00;
		
			for(int k=0;k<persons ;k++)
			{
				for(int j=0;j<realsign;j++)
				{
					double temp = findvalue(feature,allsigns[k][j],a);
				//	System.out.println("printing feature"+temp);
					if(temp>max)
					{
						max = temp; 
					}
					if(temp<min)
					{
						min = temp; 
					}
					//System.out.println("printing findvalues for verification for other person "+findvalue(feature,allsigns[k][j],a));
				}
				
			}
			//System.out.println("printing max"+max);
		//	System.out.println("printing min"+min);
			return max;
	 }
	 public static double find_min (interior_node a)
	 {
		double max =-10000.0;
		double min = 10000.0;
		
			for(int k=0;k<persons ;k++)
			{
				for(int j=0;j<realsign;j++)
				{
					double temp = findvalue(feature,allsigns[k][j],a);
					if(temp>max)
					{
						max = temp; 
					}
					if(temp<min)
					{
						min = temp; 
					}
					//System.out.println("printing findvalues for verification for other person "+findvalue(feature,allsigns[k][j],a));
				}
				
			}
			return min;
	 }
	//end of max and min functions
	public static void assignfeaturevalues() throws FileNotFoundException 
	{
		// need to initilaise them somewhere in a function . SO usnign this function
		for(int i=0;i<tempbitworand1.length;i++)
		{
		tempbitworand1[i] = 0;
		tempbitworand2[i] = 0;
		tempbitworand3[i] = 0;
		}
		 
        //creating File instance to reference text file in Java
        File text = new File("D:/Acads18/DOP18/input.txt");
     
        //Creating Scanner instnace to read File in Java
        Scanner scnr = new Scanner(text);
     
        //Reading each line of file using Scanner class
        
      
           
            for(int t=0;t<persons;t++)
            {
            	for(int j=0; j<realsign;j++)
            	{
                
	                for( int i=0;i<feature;i++)
	                {
	                	if(scnr.hasNextLine())
	                	{
	                	String line = scnr.nextLine();
	                	allsigns[t][j][i] = Double.parseDouble(line);
	                	}
	                }
            	}
            }
	}
	
	public static void copy_to_person_under_consider(int i)
	{
		for(int k=0;k<realsign;k++)
		{
		     System.arraycopy(allsigns[i][k],0,person_under_consideration[k],0,allsigns[i][k].length);
		}
	}
	public static interior_node find_initial_random_weights (interior_node a)
	{
		double lower = -1.0;//for random number generator
		double upper = 1.0;//for random number generator
		// filling bintworand1 and 2
		for(int i=0;i<feature;i++)
		{
			double random =  Math.random() * (upper - lower) + lower;
            if(random<a.alpha1[i]*a.alpha1[i])
            {
                bitworand1[i]=1;
            }
            else
            {
                bitworand1[i]=0;
            }
		}
		
		for(int i=0;i<feature;i++)
		{
			double random =  Math.random() * (upper - lower) + lower;
            if(random<a.alpha2[i]*a.alpha2[i])
            {
                bitworand2[i]=1;
            }
            else
            {
                bitworand2[i]=0;
            }
		}
		// end of filling bintworand1 and 2
		//start of filling bintworand3
		for(int i=0;i<feature;i++)
		{
			double random =  Math.random() * (upper - lower) + lower;
            if(random<a.alpha3[i]*a.alpha3[i])
            {
                bitworand3[i]=1;
            }
            else
            {
                bitworand3[i]=0;
            }
		}
		
		
		for(int i=0; i<feature;i++)
		{
			
		    a.incomingedges[i]=findthefinalrandweight(bitworand1[i], bitworand2[i]);
		  //  System.out.println("incoming now"+a.incomingedges[i]);
		    
		}
		return a;
	}
	public static interior_node calculate_for_one( interior_node a)
	{
		
		
		
		//finding the alpha by doing for some iterations
		int countactual=0;
		int countfake = 0;
		int optcountactual = 0;
		int optcountfake = 0;
		double [] tempincoming = new double[feature];
	//	for(double i=find_min(a);i<find_max(a);i=i+(find_max(a)-find_min(a))/1000)
			for(double i=1000;i>=-2000;i=i-0.5)
		{
			
			countactual = 0;
			countfake = 0;
			for(int j=0;j<realsign;j++)
			{
				//System.out.println("printing findvalues for verification for other person "+findvalue(feature,person_under_consideration[j],a)+"   estimated alpha"+i);
				if((double)findvalue(feature,person_under_consideration[j],a)<i)
				{
					countactual++;
				}
				
				
			}
			for(int k=0;k<persons;k++)
			{
				for(int j=0;j<realsign;j++)
				{//System.out.println("printing findvalues for verification for other person "+findvalue(feature,allsigns[k][j],a));
					if(findvalue(feature,allsigns[k][j],a)>i)
					{
						countfake++;
					}
					
				}
				
			}//System.out.println("This is another ifle    "+a.alpha+"   values of countactual "+countactual+"  countfake "+ countfake);
			if(countactual>optcountactual && countfake> optcountfake)
			{
				a.alpha = i;
				
				optcountactual = countactual;
				optcountfake = countfake;
				
			}
		}
		//end of the alpha finding
		for(int i=0; i<feature;i++)
		{
			
			// System.out.println("Just features    "+a.incomingedges[i]+"       ");
		    
		}
		int temp1 = optcountactual+optcountfake;
		// System.out.println("This is another ifle    "+a.alpha+"   values of countactual "+countactual+"  countfake "+ countfake);
		 System.out.println("*************** end ********************      ");
	    return a;
	}
	
	
	
	
/*	public static interior_node calculate_for_great( interior_node a)
	{
		
		
		//finding the alpha by doing for some iterations
		int countactual=0;
		int countfake = 0;
		int optcountactual = 0;
		int optcountfake = 0;
		double [] tempincoming = new double[feature];
		for(double i=-10000;i<10000;i++)
		{
			countactual = 0;
			countfake = 0;
			for(int j=0;j<realsign;j++)
			{
				if(findvalue(feature,person_under_consideration[j],a)>i)
				{
					countactual++;
				}
				//System.out.println("printing findvalues for verification for actual person "+findvalue(feature,person_under_consideration[j],a));
				
			}
			
			if(countactual >= optcountactual)
			{
				a.alphagreater = i;
				//System.out.println("i am entering this "+i+"value of count actual is "+countactual+"    "+countfake+"      "+optcountactual+"    "+optcountfake+"\n");
				optcountactual = countactual;
			
				
			}
		}
		//end of the alpha finding
	
	    return a;
	}*/
	//start of updating using the table
	public static interior_node updation( interior_node a)
	{
		int countactual=0;
		int countfake = 0;
		int optcountactual = 0;
		int optcountfake = 0;
		double degree = 0;
	    double [] optincomingedges = new double[feature];
	    double [] optincomingedges2 = new double[feature];//remove it now
	/*	double [] tempalpha1 = new double[feature];
		double [] tempalpha2 = new double[feature];
		 System.arraycopy(a.alpha1,0,tempalpha1,0,a.alpha1.length);
		 System.arraycopy(a.alpha2,0,tempalpha2,0,a.alpha2.length);
		 */
		for(int hj=0;hj<1000;hj++)
		{
			if(hj!=0)
			{
				
				//copying from above function as I did not make a separate fuction in it

				
				double lower = -1.0;//for random number generator
				double upper = 1.0;//for random number generator
				// filling bintworand1 and 2
				for(int i=0;i<feature;i++)
				{
					double random =  Math.random() * (upper - lower) + lower;
		            if(random<a.alpha1[i]*a.alpha1[i])
		            {
		                bitworand1[i]=1;
		            }
		            else
		            {
		                bitworand1[i]=0;
		            }
				}
				
				for(int i=0;i<feature;i++)
				{
					double random =  Math.random() * (upper - lower) + lower;
		            if(random<a.alpha2[i]*a.alpha2[i])
		            {
		                bitworand2[i]=1;
		            }
		            else
		            {
		                bitworand2[i]=0;
		            }
				}
				// end of filling bintworand1 and 2
				
				
				for(int i=0; i<feature;i++)
				{
					
				    a.incomingedges[i]=findthefinalrandweight(bitworand1[i], bitworand2[i])*bitworand3[i];
				    
				}
				
				
				//end of the copy from above function
				
			}
			for(int j=0;j<realsign;j++)
			{
				if(findvalue(feature,person_under_consideration[j],a)<a.alpha)
				{
					countactual++;
				}
				
			}
			for(int k=0;k<persons && k!=a.class_classifying;k++)
			{
				for(int j=0;j<realsign;j++)
				{
					if(findvalue(feature,person_under_consideration[j],a)>a.alpha)
					{
						countfake++;
					}
					
				}
			}
			
			
			if(countactual+countfake >= optcountactual+optcountfake)
			{
				optcountactual = countactual;
				optcountfake = countfake;
				
				 for(int ik=0;ik<feature;ik++)
                 {
                   
                     if(tempbitworand1[ik]==0 && bitworand1[ik]==1)
                     {
                         degree = -0.03*(3.15);
                     }
                     if(tempbitworand1[ik]==1 && bitworand1[ik]==0)
                     {
                         degree = 0.03*(3.15);
                     }
                     else 
                     {
                    	 degree = 0;
                     }
                     
                     a.alpha1[ik] = a.alpha1[ik]*(Math.sin(Math.toRadians(degree)))-Math.sqrt(1-a.alpha1[ik]*a.alpha1[ik])*(Math.cos(Math.toRadians(degree)));
                    //for alpha2
                     
                     if(tempbitworand2[ik]==0 && bitworand2[ik]==1)
                     {
                         degree = -0.03*(3.15);
                     }
                     if(tempbitworand2[ik]==1 && bitworand2[ik]==0)
                     {
                         degree = 0.03*(3.15);
                     }
                     else 
                     {
                    	 degree = 0;
                     }
                     
                     a.alpha2[ik] = a.alpha2[ik]*(Math.sin(Math.toRadians(degree)))-Math.sqrt(1-a.alpha2[ik]*a.alpha2[ik])*(Math.cos(Math.toRadians(degree)));
                     
                     //end for alpha2
                 }
					for(int i=0;i<feature;i++)
					{
					tempbitworand1[i]=bitworand1[i];
					tempbitworand2[i]=bitworand2[i];
					optincomingedges[i] = a.incomingedges[i]; 
					 optincomingedges2[i]=findthefinalrandweight(tempbitworand1[i], tempbitworand2[i])*bitworand3[i];
					}
					
				}
				 
			}
		for(int i=0;i<feature;i++)
		{
			a.incomingedges[i] = optincomingedges[i]; 
		}
		return a;
	}	
	
	
}
