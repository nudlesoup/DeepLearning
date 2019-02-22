@Author:AmeyaDhamanaskar
package qnn;


import java.util.Random;
import java.util.Vector;

public class functions {
	
	   public static Random random = new Random();
	   public static double randomInRange(double min, double max)
	    {
	        double range = max - min;
	        double scaled = random.nextDouble() * range;
	        double shifted = scaled + min;
	        return shifted; // == (rand.nextDouble() * (max-min)) + min;
	    }

	public static double findthefinalrandweight(int c, int b)
    {
		int a= (2*c)+b;
	    int multiplyfac = 10;	
        if(a == 1)
        {
            return randomInRange(-2.0,-1.0)*multiplyfac;
        }
        else if(a == 2)
        {
            return randomInRange(-1.0,0.0)*multiplyfac;
        }
        else if(a == 3)
        {
            return randomInRange(0.0,1.0)*multiplyfac;
        }
        else if(a == 4)
        {
            return randomInRange(1.0,2.0)*multiplyfac;
        }
        return 0.0;
    }
	 public static double findvalue(int numbofeature, double [] featurevalue, interior_node a)
	    {
	        double temp=0.0;
	        for (int i=0;i<numbofeature;i++)
	        {
	            temp = temp + (featurevalue[i]*a.incomingedges[i]);
	        }
	        return temp;
	       // return  1/(1+Math.exp(-temp));
	    }


}