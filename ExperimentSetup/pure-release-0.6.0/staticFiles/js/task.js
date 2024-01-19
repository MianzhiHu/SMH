/* ************************************ */
/* Define helper functions */
/* ************************************ */

/*I need to think about creating an array that will update as the experiemtn goes on. This can then be called at teh 
end of the experiment and pushed out in the csv file. I still need to test in psi turk to see about the data collection there.*/


var getStim = function() {
  //curr_stim = respStims.shift()
    curr_stim = respStims[current_trial][0]
  //var stim = base_path + curr_stim[0]
  var stim = base_path + curr_stim
  pointReward = respStims[current_trial][2]
    trialCat = respStims[current_trial][1]
  
  return '<div class = dd_stimBox><img class = dd_Stim src = ' + stim + ' </img></div>' +
    stim_response_area
}
var getInstructFeedback = function() {
    return '<div class = centerbox><p class = "center-block-text">' +
      feedback_instruct_text + '</p></div>'
}
  /* ************************************ */
  /* Define experimental variables */
  /* ************************************ */
var sumInstructTime = 0 //ms
var instructTimeThresh = 0 ///in seconds

// task specific variables
var stim_responses = ['CatA', 'CatB']


var stim_response_area = '<div class = dd_response_div>' +
  '<button class = dd_response_button id = CatA>Option A</button>' +
  '<button class = dd_response_button id = CatB>Option B</button>'

/*change these things for the blocks and change numbers*/
var base_path = 'stims/threeDintCJES/'

var stims = [['111.png',1,87],['112.png',1,84],['113.png',1,87],['114.png',1,81],['115.png',1,88],['116.png',1,75],['117.png',1,71],['118.png',1,85],['119.png',1,50],['1110.png',1,63],['1111.png',1,86],['1112.png',1,89],['1113.png',1,65],['1114.png',1,81],['1115.png',1,89],['1116.png',1,84],['1117.png',1,83],['1118.png',1,89],['1119.png',1,89],['1120.png',1,79],['1121.png',1,78],['1122.png',1,91],['1123.png',1,76],['1124.png',1,58],['1125.png',1,75],['1126.png',1,92],['1127.png',1,87],['1128.png',1,82],['1129.png',1,78],['1130.png',1,92],['1131.png',1,90],['1132.png',1,89],['1133.png',1,91],['1134.png',1,65],['1135.png',1,87],['1136.png',1,59],['1137.png',1,76],['1138.png',1,92],['1139.png',1,58],['1140.png',1,91],['1141.png',1,91],['1142.png',1,72],['1143.png',1,90],['1144.png',1,55],['1145.png',1,50],['1146.png',1,78],['1147.png',1,91],['1148.png',1,85],['1149.png',1,91],['1150.png',1,93],['1151.png',1,92],['1152.png',1,88],['1153.png',1,88],['1154.png',1,92],['1155.png',1,69],['1156.png',1,79],['1157.png',1,90],['1158.png',1,93],['1159.png',1,60],['1160.png',1,93],['1161.png',1,63],['1162.png',1,82],['1163.png',1,78],['1164.png',1,78],['1165.png',1,76],['1166.png',1,86],['1167.png',1,89],['1168.png',1,81],['1169.png',1,89],['1170.png',1,80],['1171.png',1,90],['1172.png',1,90],['1173.png',1,89],['1174.png',1,73],['1175.png',1,83],['1176.png',1,87],['1177.png',1,70],['1178.png',1,86],['1179.png',1,93],['1180.png',1,91],['1181.png',1,88],['1182.png',1,86],['1183.png',1,91],['1184.png',1,92],['1185.png',1,62],['1186.png',1,84],['1187.png',1,84],['1188.png',1,91],['1189.png',1,85],['1190.png',1,68],['1191.png',1,89],['1192.png',1,93],['1193.png',1,84],['1194.png',1,84],['1195.png',1,88],['1196.png',1,81],['1197.png',1,82],['1198.png',1,90],['1199.png',1,89],['11100.png',1,91],['11101.png',1,93],['11102.png',1,85],['11103.png',1,82],['11104.png',1,91],['11105.png',1,85],['11106.png',1,87],['11107.png',1,73],['11108.png',1,77],['11109.png',1,69],['11110.png',1,57],['11111.png',1,85],['11112.png',1,81],['11113.png',1,77],['11114.png',1,87],['11115.png',1,87],['11116.png',1,67],['11117.png',1,81],['11118.png',1,85],['11119.png',1,88],['11120.png',1,84],['11121.png',1,70],['11122.png',1,90],['11123.png',1,78],['11124.png',1,91],['11125.png',1,88],['11126.png',1,89],['11127.png',1,87],['11128.png',1,87],['11129.png',1,91],['11130.png',1,90],['11131.png',1,76],['11132.png',1,83],['11133.png',1,56],['11134.png',1,85],['11135.png',1,77],['11136.png',1,50],['11137.png',1,78],['11138.png',1,76],['11139.png',1,85],['11140.png',1,87],['11141.png',1,79],['11142.png',1,83],['11143.png',1,91],['11144.png',1,82],['11145.png',1,78],['11146.png',1,57],['11147.png',1,88],['11148.png',1,87],['11149.png',1,92],['11150.png',1,86],['11151.png',1,84],['11152.png',1,89],['11153.png',1,50],['11154.png',1,88],['11155.png',1,92],['11156.png',1,91],['11157.png',1,90],['11158.png',1,84],['11159.png',1,52],['11160.png',1,85],['11161.png',1,86],['11162.png',1,89],['11163.png',1,87],['11164.png',1,78],['11165.png',1,91],['11166.png',1,92],['11167.png',1,85],['11168.png',1,91],['11169.png',1,93],['11170.png',1,73],['11171.png',1,80],['11172.png',1,91],['11173.png',1,89],['11174.png',1,92],['11175.png',1,89],['11176.png',1,90],['11177.png',1,93],['11178.png',1,93],['11179.png',1,93],['11180.png',1,84],['11181.png',1,90],['11182.png',1,89],['11183.png',1,89],['11184.png',1,78],['11185.png',1,88],['11186.png',1,88],['11187.png',1,91],['11188.png',1,88],['11189.png',1,93],['11190.png',1,90],['11191.png',1,88],['11192.png',1,93],['11193.png',1,88],['11194.png',1,79],['11195.png',1,77],['11196.png',1,89],['11197.png',1,86],['11198.png',1,55],['11199.png',1,75],['11200.png',1,88],['11201.png',2,50],['11202.png',2,95],['11203.png',2,80],['11204.png',2,80],['11205.png',2,64],['11206.png',2,83],['11207.png',2,96],['11208.png',2,50],['11209.png',2,89],['11210.png',2,92],['11211.png',2,50],['11212.png',2,84],['11213.png',2,50],['11214.png',2,50],['11215.png',2,94],['11216.png',2,50],['11217.png',2,82],['11218.png',2,74],['11219.png',2,71],['11220.png',2,71],['11221.png',2,94],['11222.png',2,50],['11223.png',2,93],['11224.png',2,79],['11225.png',2,96],['11226.png',2,90],['11227.png',2,50],['11228.png',2,50],['11229.png',2,64],['11230.png',2,75],['11231.png',2,60],['11232.png',2,79],['11233.png',2,97],['11234.png',2,71],['11235.png',2,73],['11236.png',2,80],['11237.png',2,80],['11238.png',2,97],['11239.png',2,72],['11240.png',2,94],['11241.png',2,50],['11242.png',2,82],['11243.png',2,95],['11244.png',2,66],['11245.png',2,88],['11246.png',2,90],['11247.png',2,57],['11248.png',2,96],['11249.png',2,50],['11250.png',2,50],['11251.png',2,74],['11252.png',2,91],['11253.png',2,72],['11254.png',2,50],['11255.png',2,67],['11256.png',2,88],['11257.png',2,79],['11258.png',2,92],['11259.png',2,51],['11260.png',2,63],['11261.png',2,73],['11262.png',2,96],['11263.png',2,62],['11264.png',2,57],['11265.png',2,86],['11266.png',2,95],['11267.png',2,50],['11268.png',2,64],['11269.png',2,82],['11270.png',2,97],['11271.png',2,89],['11272.png',2,53],['11273.png',2,97],['11274.png',2,76],['11275.png',2,63],['11276.png',2,80],['11277.png',2,80],['11278.png',2,77],['11279.png',2,50],['11280.png',2,87],['11281.png',2,87],['11282.png',2,80],['11283.png',2,91],['11284.png',2,63],['11285.png',2,85],['11286.png',2,59],['11287.png',2,72],['11288.png',2,97],['11289.png',2,74],['11290.png',2,81],['11291.png',2,70],['11292.png',2,72],['11293.png',2,75],['11294.png',2,91],['11295.png',2,85],['11296.png',2,78],['11297.png',2,79],['11298.png',2,94],['11299.png',2,80],['11300.png',2,92],['11301.png',2,77],['11302.png',2,79],['11303.png',2,74],['11304.png',2,85],['11305.png',2,91],['11306.png',2,74],['11307.png',2,65],['11308.png',2,63],['11309.png',2,50],['11310.png',2,50],['11311.png',2,50],['11312.png',2,86],['11313.png',2,81],['11314.png',2,77],['11315.png',2,97],['11316.png',2,69],['11317.png',2,59],['11318.png',2,80],['11319.png',2,90],['11320.png',2,73],['11321.png',2,98],['11322.png',2,97],['11323.png',2,96],['11324.png',2,97],['11325.png',2,50],['11326.png',2,50],['11327.png',2,87],['11328.png',2,63],['11329.png',2,84],['11330.png',2,69],['11331.png',2,57],['11332.png',2,53],['11333.png',2,81],['11334.png',2,84],['11335.png',2,92],['11336.png',2,66],['11337.png',2,96],['11338.png',2,75],['11339.png',2,58],['11340.png',2,63],['11341.png',2,79],['11342.png',2,98],['11343.png',2,70],['11344.png',2,50],['11345.png',2,67],['11346.png',2,68],['11347.png',2,67],['11348.png',2,70],['11349.png',2,53],['11350.png',2,88],['11351.png',2,82],['11352.png',2,78],['11353.png',2,50],['11354.png',2,50],['11355.png',2,72],['11356.png',2,93],['11357.png',2,55],['11358.png',2,77],['11359.png',2,72],['11360.png',2,94],['11361.png',2,97],['11362.png',2,95],['11363.png',2,85],['11364.png',2,91],['11365.png',2,67],['11366.png',2,58],['11367.png',2,62],['11368.png',2,78],['11369.png',2,71],['11370.png',2,75],['11371.png',2,76],['11372.png',2,53],['11373.png',2,83],['11374.png',2,73],['11375.png',2,96],['11376.png',2,94],['11377.png',2,79],['11378.png',2,50],['11379.png',2,53],['11380.png',2,88],['11381.png',2,79],['11382.png',2,74],['11383.png',2,80],['11384.png',2,81],['11385.png',2,56],['11386.png',2,94],['11387.png',2,71],['11388.png',2,50],['11389.png',2,94],['11390.png',2,65],['11391.png',2,57],['11392.png',2,80],['11393.png',2,81],['11394.png',2,81],['11395.png',2,94],['11396.png',2,76],['11397.png',2,92],['11398.png',2,79],['11399.png',2,91],['11400.png',2,60]]


var images = []
for (var i = 0; i < stims.length; i++) {
  images.push(base_path + (stims[i][0]))
}
//preload images
jsPsych.pluginAPI.preloadImages(images)

var current_trial = 0
//var respStims = jsPsych.randomization.shuffle(stims)
var respStims = jsPsych.randomization.shuffle(stims)
var reference_stim = ''
var curr_stim = ''
var pointReward = []
var trialCat = []
var stim_ratings = {}
//var stim_feed = []
var pointsTotal = 0
//var correctBlock = 0/4
var correctTotal = 0
var correctTotal1 = 0
var correctTotal2 = 0
var correctTotal3 = 0
var correctTotal4 = 0
var blockPercent = 0
/*for (var s = 0; s < stims.length; s++) {
  stim_ratings[stims[s]] = {}
}*/

for (var s = 0; s < stims.length; s++) {
  stim_ratings[stims[s]] = {}
}
/* ************************************ */
/* Set up jsPsych blocks */
/* ************************************ */

//Set up post task questionnaire
var post_task_block = {
   type: 'survey-text',
   data: {
       trial_id: "post task questions"
   },
   questions: ['<p class = center-block-text style = "font-size: 20px">Please summarize what you were asked to do in this task.</p>',
              '<p class = center-block-text style = "font-size: 20px">Do you have any comments about this task?</p>'],
   rows: [15, 15],
   columns: [60,60]
};

/* define static blocks */
var end_block = {
  type: 'poldrack-text',
  timing_response: 180000,
  data: {
    trial_id: 'end',
    exp_id: 'threeDintMT'
  },
  text: '<div class = centerbox><p class = "center-block-text">Thanks for completing this task!</p><p class = "center-block-text">Press <strong>enter</strong> to continue.</p></div>',
  cont_key: [13],
  timing_post_trial: 0,
  /*on_finish: assessPerformance*/
};

var feedback_instruct_text =
  'Welcome to the experiment. This task will take around 45 minutes. Press <strong>enter</strong> to begin.'
var feedback_instruct_block = {
  type: 'poldrack-text',
  data: {
    trial_id: 'instruction'
  },
  cont_key: [13],
  text: getInstructFeedback,
  timing_post_trial: 0,
  timing_response: 180000
};
/// This ensures that the subject does not read through the instructions too quickly.  If they do it too quickly, then we will go over the loop again.
var instructions_block = {
  type: 'poldrack-instructions',
  data: {
    trial_id: 'instruction'
  },
  pages: [
    "<div class = centerbox><p class = 'block-text'>In this task you will see a variety of lines on the screen. Some lines may be more rewarding than others. Over the course of the experiment, you may learn how to maximize the rewards you receive.</p></div>"
  ],
  allow_keys: false,
  show_clickable_nav: true,
  //timing_post_trial: 1000
};

var instruction_node = {
  timeline: [feedback_instruct_block, instructions_block],
  /* This function defines stopping criteria */
  loop_function: function(data) {
    for (i = 0; i < data.length; i++) {
      if ((data[i].trial_type == 'poldrack-instructions') && (data[i].rt != -1)) {
        rt = data[i].rt
        sumInstructTime = sumInstructTime + rt
      }
    }
    if (sumInstructTime <= instructTimeThresh * 1000) {
      feedback_instruct_text =
        'Read through instructions too quickly.  Please take your time and make sure you understand the instructions.  Press <strong>enter</strong> to continue.'
      return true
    } else if (sumInstructTime > instructTimeThresh * 1000) {
      feedback_instruct_text =
        'Done with instructions. Press <strong>enter</strong> to continue.'
      return false
    }
  }
   
}

var resp_block = {
  type: 'single-stim-button',
  // stimulus: getStim,
  stimulus: getStim,
  button_class: 'dd_response_button',
  data: {
    trial_id: 'stim',
    exp_stage: 'response_rating'
  },
  timing_stim: 10000,
  timing_response: 10000,
  response_ends_trial: true,
  timing_post_trial: 500,
  on_finish: function(data) {
    var numeric_rating = stim_responses.indexOf(data.mouse_click)
    if (data.mouse_click === -1) {
      numeric_rating = 'NaN'
    } 
    jsPsych.data.addDataToLastTrial({
      //'stim': curr_stim[0].slice(0, -4),
        'stim': curr_stim,//.slice(0, -4),
      'coded_response': numeric_rating,
      'trial_num': current_trial
    })
    //stim_ratings[curr_stim].response = numeric_rating
      stim_ratings[respStims[current_trial]].response = numeric_rating
     //current_trial += 1  
  }
}
    

var get_feedback = function() {
    var trialfeed = jsPsych.data.getLastTrialData()
	var stim_feed = stim_responses.indexOf(trialfeed.mouse_click)
    var points = 0
    var correct = 0
    if (current_trial == 100){
        correctTotal1 = correctTotal
        correctTotal = 0
    } else if (current_trial == 200){
        correctTotal2 = correctTotal
        correctTotal = 0
    } else if (current_trial == 300){
        correctTotal3 = correctTotal
        correctTotal = 0
    } else if (current_trial == 400){
        correctTotal4 = correctTotal
        correctTotal = 0
    }

	if (stim_feed == -1) {
        points = 0
        correct = 0
        return "<div class = centerbox><div class = center-text>" +
			"Please respond faster!</div>"
	} else if (stim_feed == 0) {
        //if (stim_feed[current_trial] == (curr_stim[1] - 1)) {
            if (stim_feed == (trialCat-1)) {
            //FB = curr_stim
            points = pointReward
            correct = 1
                correctTotal = correctTotal + correct
            return "<div class = centerbox><div class = center-text>" +
                "" + points + "</div></div>"
        //} else if (stim_feed[current_trial] != (curr_stim[1] - 1)){
            } else if (stim_feed != (trialCat-1)){
            //FB = curr_stim
            points = 0
            correct = 0
                correctTotal = correctTotal + correct
            return "<div class = centerbox><div class = center-text>" +
                "Incorrect: 0 Points</div></div>"
        }
    } else if (stim_feed == 1) {
        //if (stim_feed[current_trial] == (curr_stim[1] - 1)) {
            if (stim_feed == (trialCat - 1)) {
            //FB = curr_stim
            points = pointReward
            correct = 1
                correctTotal = correctTotal + correct
            return "<div class = centerbox><div class = center-text>" +
                "" + points + "</div></div>"
        //} else if (stim_feed[current_trial] != (curr_stim[1] - 1)){
            } else if (stim_feed != (trialCat - 1)){
            //FB = curr_stim
            points = 0
            correct = 0
                correctTotal = correctTotal + correct
            return "<div class = centerbox><div class = center-text>" +
                "Incorrect: 0 Points</div></div>"
        }
    } 
}

var FB_stage = {
	type: "poldrack-single-stim",
	data: {
		trial_id: 'feedback_stage'
	},
	stimulus: get_feedback,
	is_html: true,
	choices: 'none',
	timing_response: 1000,
	continue_after_response: false,
	timing_post_trial: 0,
	on_finish: function() {
		//jsPsych.data.addDataToLastTrial({
			//'coded_response': current_trial,
		//})
        current_trial = current_trial + 1
	}
}

var get_blockfeed = function() {
    blockPercent = correctTotal/100
    return '<div class = centerbox><p class = "center-block-text"> You have completed ' +
      current_trial + ' trials in total. You were rewarded  ' + blockPercent + '% of the time in the past 100 trials. Press <strong>enter</strong> to continue.</p></div>'
}
var respFeed1 = {
  type: 'poldrack-text',
  data: {
    trial_id: 'instruction'
  },
  cont_key: [13],
  text: get_blockfeed,
  timing_post_trial: 0,
  timing_response: 180000
}

/* create experiment definition array */
var threeDintMT_experiment = [];

threeDintMT_experiment.push(instruction_node);
/*if (Math.random() < 0.5) {*/
//threeDintMT_experiment.push(start_stim_block);
/*for (var i = 0; i < 400; i++) {
        threeDintMT_experiment.push(resp_block);
        threeDintMT_experiment.push(FB_stage);
    if (i == 4){
        threeDintMT_experiment.push(respFeed1);
    } else if (i == 99){
        threeDintMT_experiment.push(respFeed1);
    } else if (i == 199){
        threeDintMT_experiment.push(respFeed1);
    } else if (i == 299){
        threeDintMT_experiment.push(respFeed1);
   //} if (i == 5){
       //threeDintMT_experiment.push(respFeed1);//for testing
    }
    
} */
//for testing
for (var i = 0; i < 20; i++) {
        threeDintMT_experiment.push(resp_block);
        threeDintMT_experiment.push(FB_stage);
    if (i == 4){
        threeDintMT_experiment.push(respFeed1);
    } else if (i == 9){
        threeDintMT_experiment.push(respFeed1);
    } else if (i == 14){
        threeDintMT_experiment.push(respFeed1);
    } else if (i == 19){
        threeDintMT_experiment.push(respFeed1);
   //} if (i == 5){
       //threeDintMT_experiment.push(respFeed1);//for testing
    }
    
} 
/*threeDintMT_experiment.push(setup_block);*/
threeDintMT_experiment.push(post_task_block);
threeDintMT_experiment.push(end_block);