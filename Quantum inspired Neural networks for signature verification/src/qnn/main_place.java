@Author:AmeyaDhamanaskar
package qnn;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.io.*;



public class main_place extends taking_input {
static int allcountyes=0;
static int allcountno=0;
static double[] features_for_testing = new double [45];
	public static void taking_test_input_from_file( String a ) throws FileNotFoundException 
	{
		 
		
	    //creating File instance to reference text file in Java
	    File text = new File(a);
	 
	    //Creating Scanner instnace to read File in Java
	    Scanner scnr = new Scanner(text);
	 
	    //Reading each line of file using Scanner class
	    int lineNumber = 1;
	  
	       
	        for(int t=0;t<45;t++)
	        {
	        	if(scnr.hasNextLine())
            	{
            	String line = scnr.nextLine();
            	features_for_testing[t] = Double.parseDouble(line);
            	}
	        	
	        }
	}
	//end iof taking input for testing function
	
	
	public static int check(int [] array)
	{
		
		for(int i=0;i<array.length;i++)
		{
			if(array[i]!=1)
			{
				return 0;
			}
		}
		return 1;
	}
	public static void main(String[] args) throws IOException
	{
	 int max = 0;
	 Vector<interior_node> opt = new Vector<interior_node>(1);
	  int [] testpositive = new int[36];
	  int [] opttestpositive = new int [36];
	  int [][] testnegative = new int[36][2]; 
	  int myfate = 0;
	 
		for(int accu = 0; accu < 1;accu++){	
			allcountyes=0;
			for(int yi=0;yi<36;yi++){
				testpositive[yi]=0;
			}
			 for(int ui=0;ui<36;ui++)
				{
					for(int uj=0;uj<2;uj++)
					testnegative[ui][uj]=0;
				}
		Vector<interior_node> vec = new Vector<interior_node>(1);	
	 //   interior_node [] my = new interior_node [10];
	    
	   int class_under=0;
	  int in_class=0;
	  int fe = 45;
	//  taking_input t = new taking_input ();
	  int [][] classifies = new int [persons][realsign];
	  
	  for(int i=0;i< persons;i++)
	  {
		  for(int j=0;j<realsign;j++)
		  classifies[i][j]=-1;
	  }
	  while(class_under!=persons-1)
	  {
		  interior_node a = new interior_node();
	    a.class_classifying = class_under;
		a.number_within_class=in_class;
		for(int i=0; i<fe;i++)
		{
		a.alpha1[i]=0.707;//
		a.alpha2[i]=0.707;//
		a.alpha3[i]=0.707;
		a.alpha = -200000.5;
		}
		
		assignfeaturevalues();
		copy_to_person_under_consider(class_under);
		a = find_initial_random_weights (a);
	//	a= calculate_for_great(a);
		int countactual = 0;
		int countfake = 0;
		int optcountactual = 0;
		int optcountfake = 0;
		interior_node b = new interior_node();
		b=calculate_for_one(a);
		b=updation(a);
		for(int i=0;i<5;i++)
		{
			System.out.println("Now class "+class_under+" is classified");
			b=calculate_for_one(b);
			b=updation(b);
		//b= calculate_for_great(a);
		countactual = 0;
		countfake = 0;
		for(int j=0;j<realsign;j++)
		{
			if(findvalue(feature,person_under_consideration[j],b)<b.alpha)
			{
				countactual++;
			}
			//System.out.println("printing findvalues for verification for actual person "+findvalue(feature,person_under_consideration[j],a));
			
		}
		for(int k=0;k<persons && k!=class_under ;k++)
		{
			for(int j=0;j<realsign;j++)
			{
				if(findvalue(feature,allsigns[k][j],b)>b.alpha)
				{
					countfake++;
				}
				//System.out.println("printing findvalues for verification for other person "+findvalue(feature,allsigns[k][j],a));
			}
			
		}
		if(countactual>optcountactual && countfake>optcountfake)
		{
			a=b;
			//System.out.println("i am entering this "+i+"value of count actual is "+countactual+"    "+countfake+"      "+optcountactual+"    "+optcountfake+"\n");
			optcountactual = countactual;
			optcountfake = countfake;
			
		}
		
		}
	
		
		
		vec.add(a);// inerting it into the vector
	//	my[sillycount++]=a;
		/*for(int i=0;i<45;i++)
		{
			System.out.println("newly added" + vec.get(vec.size()-1).incomingedges[i]);
		}*/
	
		 
		
		int flag1= 0;
		for(int i = 0; i<realsign;i++)
		{
			
			if(findvalue(fe,person_under_consideration[i],a)<a.alpha)
			{
				//if(classifies[class_under][i]!=1){flag1 = 1;}
				classifies[class_under][i]=1;
			}
		}
		
		if(check(classifies[class_under])==1)
		{
			class_under++;
			in_class = 0;
			System.out.println("Now class "+class_under+" is classified");
			
		}
		else
		{
			in_class++;
		}
	}
	  //remove
	  for(int i=0;i<vec.size();i++)
	  {
		  System.out.println("This is removed "+"    "+vec.get(i).alpha+"       "+vec.get(i).class_classifying);
	  }
	  //
	
	 
	//testing the signatures int max = 0;
	  
	
	/*  for(int yu=0;yu<100;yu++){
			 allcountyes = 0;*/
	  //for(int pj=0;pj<35;pj++)
	  for(int pj=0;pj<7;pj++)
		{
		  for(int pk=0;pk<2;pk++)
			//for(int pk=0;pk<2;pk++)
			{
				String ai = "D:/Acads18/DOP18/attachments/";
				String b = Integer.toString(pj);
				String c = Integer.toString(pk);
				String d = ai.concat(b);
				d = d.concat(c);
				d= d.concat(".txt");
			taking_test_input_from_file(d);
	 int flag=0;
	  
	/*  int class_to_be_checked;
	  Scanner in = new Scanner(System.in);
	  
    System.out.println("Enter a string");
    class_to_be_checked = in.nextInt();*/
	
	 
	  for(int i=0;i<vec.size();i++)
	  {
		  
		  if(vec.get(i).class_classifying==pj )
		  {
			  System.out.println("This is class "+findvalue(feature,features_for_testing,vec.get(i))+"    "+vec.get(i).alpha+"       "+vec.get(i).alphagreater);
			  
				  if( findvalue(feature,features_for_testing,vec.get(i))>vec.get(i).alpha)
				  {
					 //System.out.println("Congrats. This belongs to the same class");
					 // System.exit(0);
					  allcountyes++;
					  testpositive[pj]++;
					  testnegative[pj][pk] = -1;
					  System.out.println("This is good class "+pj+"       "+ allcountyes);
					  flag = 1;
					  
					 break;
				  }
				  
			  
		  }
		//  else if(vec.get(i-1).c)
		  
	  }
	 
	/*  for(int i=0;i<vec.size();i++)
	  {
		  
		  System.out.println("in the main function printing size class and within_class"+vec.size()+"       "+vec.get(i).class_classifying+"     "+vec.get(i).number_within_class+"      "+vec.get(i).alpha+"\n");
		   
	  }
	  for(int i=0;i<sillycount;i++)
	  {
		  System.out.println("second and within_class"+my[i].alpha+"\n");
	  }*/
	 // System.out.println("Sorry. This is not from the same class");
	  //end of thesting
	
	}
	}
	 System.out.println("for checking allcounts"+allcountyes);
	  if(max<allcountyes)
	  {
		  max = allcountyes;
		  System.arraycopy(testpositive, 0, opttestpositive, 0, testpositive.length);
	  }
	  
	  
	 
	  
	  //output to a file
	  File file = new File("D:/Acads18/DOP18/results/abc.txt");
	  if (!file.exists()) {
			file.createNewFile();
		}
	  FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
	  for(int i=0;i<36;i++)
	  {
		  System.out.println("This is the final matrix "+opttestpositive[i]+" for class "+i);
		  //String content = Integer.toString(opttestpositive[i]);
		//bw.write(content);
		  bw.write('\n');
		  
	  }
	  bw.write("This is next");
	  for(int i=0;i<36;i++)
	  {
		 System.out.println("This is the final matrix "+opttestpositive[i]+" for class "+i);
		  //String content = Integer.toString(3-opttestpositive[i]);
		 //bw.write(content);
		  bw.write('\n');
		  
	  }
	  bw.flush();
	  bw.close();

		}//end of acc loop
		 if(max<allcountyes)
		  {
			  max = allcountyes;
			  System.arraycopy(testpositive, 0, opttestpositive, 0, testpositive.length);
			 
			//  opt=(Vector)vec.clone();
			  
		  }
		 for(int i=0;i<opttestpositive.length;i++) {
			 //System.out.print(opttestpositive[i]);
		 }
	//System.out.println("This is out of 7     "+ max);
	//System.out.println("This is out of 7     "+ opt.size());
	for(int ui=0;ui<7;ui++)//changed from 35
	{
		for(int uj=0;uj<2;uj++)
		{
		if(testnegative[ui][uj]==0){
			myfate++;
			System.out.print("1");
		}
		else
			System.out.print("0");
		}
	}
	System.out.println(" classified : "+ myfate);
	//putting the interior_node in a file for further use
	 File file = new File("D:/Acads18/DOP18/results/xyz.txt");
	  if (!file.exists()) {
			file.createNewFile();
		}
	  FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
	  for(int i=0;i<opt.size();i++)
	  {
		  String classi = Integer.toString(opt.get(i).class_classifying);
			 String classip = Integer.toString(opt.get(i).number_within_class);
			 String alphip = Double.toString(opt.get(i).alpha);
			 bw.write(classi);
			  bw.write('\n'); 
			  bw.write(classip);
			  bw.write('\n');
			  bw.write(alphip);
			  bw.write('\n');
		  for(int j=0; j<45;j++)
		  {
		 String content = Double.toString(opt.get(i).incomingedges[j]);
		  bw.write(content);
		  bw.write('\n');
		  
		  }
	 }
	  bw.flush();
	  bw.close();
	}
}