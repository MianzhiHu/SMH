// Experiment javascript code


// Matrices to hold experiment data
var timeline = []; // Will hold the trial order
var dataMatrix = []; // Holds javascript key press numbers to determine what option was choose in the previous trial
var rewardMatrix = []; // Holds the reward value for each trial
var choiceMatrix = []; // Holds the converted option numbers for each trial
var bestMatrix = []; // 1 and 0 for best choice made on each trial
var bonusRewardTotal = [];
var reaction = [];

// Allows use of some math tools for distributions
new p5()

// Define task settings
// Study - 0,1
// Rwd - 0,1
// Var - 0,1,2
// Freq - 0,1

// Study 1a-1 - 0,0,0,0
// Study 1a-2 - 0,0,1,0
// Study 1a-3 - 0,0,2,0

// Study 1b - 1,0,0,1
// Study 1b - 1,0,1,1
// Study 1b - 1,0,2,1


var StudyNum = 1
var RwdType = 0
var VarType = 1
var FreqType = 1

// Experiment parameters and arrays
var choiceSeen = 0
var trialnum = 0;
var rewardTotal = 0;
var trainShow = 0
var testShow = 0
var sTime = 0
var eTime = 0

// Set up data arrays
var stimuli = []
var leftStim = []
var rightStim = []
var choseStim = []
var currPhase = ''
var nextFeed = ''

// How many blocks?
var trainBlockNum = 5
var testBlockNum = 5


// Define reward values
const RwdMat = [[.65,.35,.75,.25],[.65,.35,1,1]]

// Define variance
var VarMat = [[[.12,.24,.48],[.12,.12,.28]],
  [[.12,.24,.48],[.12,.12,.28]],
  [[.11,.22,.43],[.43,0,.28]],
  [[.11,.22,.43],[.43,0,.28]]]
  console.log(VarMat)

// Define frequency, and number of blocks per trial type
var FreqMat = [[20,10],[15,15]]

// Get reward values
var RwdA = RwdMat[RwdType][0]
var RwdB = RwdMat[RwdType][1]
var RwdC = RwdMat[RwdType][2]
var RwdD = RwdMat[RwdType][3]

// get variance values
var VarA = VarMat[0][StudyNum][VarType]
var VarB = VarMat[1][StudyNum][VarType]
var VarC = VarMat[2][StudyNum][VarType]
var VarD = VarMat[3][StudyNum][VarType]

var Freq = FreqMat[FreqType]


// Array to hold all of the stimuli pictures
var stimArray = [
  ['static/stims/fractal1.jpg', 'static/stims/fractal1select.png'],
  ['static/stims/fractal2.jpg', 'static/stims/fractal2select.png'],
  ['static/stims/fractal3.jpg', 'static/stims/fractal3select.png'],
  ['static/stims/fractal4.jpg', 'static/stims/fractal4select.png'],
  ['static/stims/fractal5.jpg', 'static/stims/fractal5select.png'],
  ['static/stims/fractal6.jpg', 'static/stims/fractal6select.png'],
  ['static/stims/fractal7.jpg', 'static/stims/fractal7select.png'],
  ['static/stims/fractal8.jpg', 'static/stims/fractal8select.png'],
  ['static/stims/fractal9.jpg', 'static/stims/fractal9select.png'],
  ['static/stims/fractal10.jpg', 'static/stims/fractal10select.png'],
  ['static/stims/fractal11.jpg', 'static/stims/fractal11select.png'],
  ['static/stims/fractal12.jpg', 'static/stims/fractal12select.png']
];

// randomize and select four images
var stimShuffled = jsPsych.randomization.shuffle(stimArray);
var stims = stimShuffled.slice(0, 5);
console.log(stims)
jsPsych.pluginAPI.preloadImages(stims);


// Determine order of AB -CD Pairings
var ABFirst = Math.floor(Math.random() * 100);
var CDFirst = Math.floor(Math.random() * 100);
var ABOrderInit = [1, 2];
var CDOrderInit = [3, 4];
var ABOrder = jsPsych.randomization.shuffle(ABOrderInit);
var CDOrder = jsPsych.randomization.shuffle(CDOrderInit);
if (ABFirst >= CDFirst) {
  var orderMatrix = ABOrder.concat(CDOrder);
} else {
  var orderMatrix = CDOrder.concat(ABOrder);
}



// Add all stim information into objects
var stimA={
  stimulus:stims[0][0],
  reward:RwdA,
  variance:VarA,
  option:orderMatrix[0],
  optionText:'Option A',
  optionTextLoc:'centeredText1',
  optionLoc:'centered1',
  optionKey:'a',
  rewardTextLoc:'rwdText1',
  stimulusChose:stims[0][1],
  stimulusChoseLoc:'centered1chose'
}
var stimB={
  stimulus:stims[1][0],
  reward:RwdB,
  variance:VarB,
  option:orderMatrix[1],
  optionText:'Option S',
  optionTextLoc:'centeredText2',
  optionLoc:'centered2',
  optionKey:'s',
  rewardTextLoc:'rwdText2',
  stimulusChose:stims[1][1],
  stimulusChoseLoc:'centered2chose'
}
var stimC={
  stimulus:stims[2][0],
  reward:RwdC,
  variance:VarC,
  option:orderMatrix[2],
  optionText:'Option K',
  optionTextLoc:'centeredText3',
  optionLoc:'centered3',
  optionKey:'k',
  rewardTextLoc:'rwdText3',
  stimulusChose:stims[2][1],
  stimulusChoseLoc:'centered3chose'
}
var stimD={
  stimulus:stims[3][0],
  reward:RwdD,
  variance:VarD,
  option:orderMatrix[3],
  optionText:'Option L',
  optionTextLoc:'centeredText4',
  optionLoc:'centered4',
  optionKey:'l',
  rewardTextLoc:'rwdText4',
  stimulusChose:stims[3][1],
  stimulusChoseLoc:'centered4chose'
}

// Add all objects to an array and randomize them
var stimMatrix = [stimA,stimB,stimC,stimD]

var optionArrange = []
for (var i = 0; i < 4; i++){
  for (var j = 0; j < 4; j++){
    if (stimMatrix[j].option == i+1){
      optionArrange[i] = stimMatrix[j]
    }
  }
}


// This array will reference later to show the correct combination of options onscreen.
var trialOrder = [[0,1],[2,3],[0,2],[1,2],[0,3],[1,3]]

// Create Random Train Trial Order
var ABtrial = Array(Freq[0]).fill(0);
var CDtrial = Array(Freq[1]).fill(1);
var trainTrialMatrix = ABtrial.concat(CDtrial);
var trainTrials1 = [];
var trainTrials2 = [];
for (let i = 0; i < trainBlockNum; i++) {
  var trainBlock1 = jsPsych.randomization.shuffle(trainTrialMatrix);
  trainTrials1 = trainTrials1.concat(trainBlock1)
  var trainBlock2 = jsPsych.randomization.shuffle(trainTrialMatrix);
  trainTrials2 = trainTrials2.concat(trainBlock2)
}

//  Create Random Transfer Trial Order
var ACtrial = Array(5).fill(2);
var BCtrial = Array(5).fill(3);
var ADtrial = Array(5).fill(4);
var BDtrial = Array(5).fill(5);
var transferTrialMatrix = ACtrial.concat(BCtrial, ADtrial, BDtrial);
var transferTrials1 = [];
var transferTrials2 = [];
for (let i = 0; i < testBlockNum; i++) {
  var transferBlock1 = jsPsych.randomization.shuffle(transferTrialMatrix);
  transferTrials1 = transferTrials1.concat(transferBlock1)
  var transferBlock2 = jsPsych.randomization.shuffle(transferTrialMatrix);
  transferTrials2 = transferTrials2.concat(transferBlock2)
}


//All trial matrix
var allTrials1 = trainTrials1.concat(transferTrials1)
var allTrials2 = trainTrials2.concat(transferTrials2)
var allTrials = allTrials1.concat(allTrials2)


// Get the right simuli based on trial order and trial type
var getStim = function() {

      // Grab both stimuli
      stimuli = [optionArrange[trialOrder[(allTrials[trialnum])][0]],optionArrange[trialOrder[(allTrials[trialnum])][1]]]
      leftStim = stimuli[0]
      rightStim = stimuli[1]
      choseStim = stimuli

      var rwdTxt = rewardTotal
      var totalRwd = rewardTotal

      // Is this test or training? Show points total or not
      showPoint = ''
      if (trainShow == 1){
        showPoint = "<div id ='bankPos'>Total: "+totalRwd.toFixed(2)+"</div>"
      }
      if (trainShow == 0){
        showPoint = ""
      }

      // Return the right labels
      return "<div class='"+leftStim.optionTextLoc+"'>" + leftStim.optionText + "</div>"+
      "<div class='"+rightStim.optionTextLoc+"'>" + rightStim.optionText + "</div>"+
      showPoint+
      "<div class = 'promptText'><p>Please choose an option by clicking on one of the boxes above...</p></div>"
}


// function specifically for making the below animation wait a bit.
function wait(ms) {
  var d = new Date();
  var d2 = null;
  do { d2 = new Date(); }
  while(d2-d < ms);
}

// Get the right stimuli images to show and use as buttons
var getButton = function() {
    return ["<img id='"+leftStim.optionLoc+"' src= '" + leftStim.stimulus + "'></img>","<img id='"+rightStim.optionLoc+"' src= '" + rightStim.stimulus + "'></img>"]
}

// This function opens up the right keys for responses
var getChoices = function() {

    return [leftStim.optionKey,rightStim.optionKey]
}


// function to figure reward receipt
var getReward = function(a) { //this function takes an argument

    // get the right reward info
    choseOpt = choseStim[a]
    sdVal = Math.sqrt(choseOpt.variance)
    rwdVal = choseOpt.reward

    // Gets the random gaussian reward value.
    trialReward = randomGaussian(rwdVal,sdVal)

    var numRand = Math.random()
    if (StudyNum==1 && VarType==1){
      if (numRand<=rwdVal){
        trialReward = 1
      } else {
        trialReward = 0
      }
    }

    // we use the push command to send the value an array. Appends the value.
    rewardMatrix.push(trialReward)
    return trialReward

}


// Function to show proper feedback screen with reward shown
var getRewardFeedback = function() {

  // Get the right reward info form the previous function
  var testMat = dataMatrix[dataMatrix.length - 1]
  var rewardVal = getReward(testMat)
  rewardTotal = rewardTotal + rewardVal
  var textRwdTotal = rewardTotal

  // Set it to show two decimals
  var txt = rewardVal.toFixed(2)

  // call in stim info
  var choseOpt = choseStim[testMat]
  choiceMatrix.push(choseOpt.option);

  // Show the reward info or just the image selection? Training or test?
  // The nextFeed var gets the actual feedback info reaady for display. The return shows gives the info needed for the selection animation.
  if (testShow == 0){
    nextFeed = "<div id = '"+choseOpt.stimulusChoseLoc+"'><div id='"+choseOpt.rewardTextLoc+"'><p style='margin:0;padding:0;'>+"+txt+"</p><p style='margin:0;padding:0;'>Total: "+textRwdTotal.toFixed(2)+"</p></div></div>"+
                "<div class='"+choseOpt.optionTextLoc+"'>"+choseOpt.optionText+"</div>"
    return "<img id='"+choseOpt.optionLoc+"' src='"+choseOpt.stimulus+"'></div><div id='"+choseOpt.stimulusChoseLoc+"'></div><div class='"+choseOpt.optionTextLoc+"'>"+choseOpt.optionText+"</div>"
  } else if (testShow == 1){
    nextFeed = "<img id='"+choseOpt.optionLoc+"' src='"+choseOpt.stimulusChose+"'></div><div class='"+choseOpt.optionTextLoc+"'>"+choseOpt.optionText+"</div>"
    return "<img id='"+choseOpt.optionLoc+"' src='"+choseOpt.stimulusChose+"'></div><div class='"+choseOpt.optionTextLoc+"'>"+choseOpt.optionText+"</div>"
  }

}


// funciton used to figure if best choice was chose, and save local data if needed
var getDataVal = function() {
  trialnum = trialnum + 1 //Be careful when incrementing. Check to make sure that the data you might save doesnt call the wrong trial numeber
  return {
    reward: rewardMatrix[rewardMatrix.length - 1],
    keyResponse: choiceMatrix[choiceMatrix.length - 1],
    bank: rewardTotal,
    react: reaction[reaction.length-1],
    trialPhase: currPhase,
    trialType: allTrials[trialnum-1]+1,
    trialNumber: trialnum-1,
    Save_Data: 'Save'
  }
}


// Animation for the boxes. If you use this, be careful, the wrong var uses can crash the browser with never ending looping.
var boxAnim = function() {
  var picked = dataMatrix[dataMatrix.length-1];

  var choseOpt = choseStim[picked]
  var choseAnim = choseOpt.stimulusChoseLoc

  if (testShow == 0){
    var boxChange = document.getElementById(choseAnim);
    var id = setInterval(frame, 15);
    var animCount = 0
    function frame() {

      // This is where stuff can crash.
      if (animCount == 15) {
        boxChange.style.opacity = '1';
        clearInterval(id);
      } else if (animCount % 2 == 0) {
        boxChange.style.opacity = '.5';
        wait(75)
        animCount++
      } else {
        boxChange.style.opacity = '1';
        wait(75)
        animCount++
      }

    }
  }
}

// Get the premade trial image we made earlier.
var getNextFeed = function(){
  return nextFeed
}


// // Figures bonus amount if bonus trials were seen.
var sumPoint = function() {

  return "<p>You have completed the experiment! Thank you for your time.</p>"+
  "<p></p>"+//"<p>Since you successfully completed the study, we are giving you an additional bonus of $6.00.</p>"+
  "<p>Thank you for your time in completing this experiment. Over the course of this experiment you have repeatedly chosen between multiple options. "+
  "Each of these option pairings had a different mean reward value and the indiviudal rewards that you received varied by a certain degree around this mean value. In "+
  "the experiment today, we were wanting to see how differing levels of uncertainty of reward values influence choice. Once again, thank you for your time.</p>"+
  "<p></p>"+
  "<p>Press the button below to exit and receive a survey code...</p>"
}


var getTestInstrux = function(){

  return "<p>You have completed the first portion!"+
    "<p></p>"+
    "<p>In this phase, you will again be shown four options to choose from.</p>"+
    "<p>However, this time, the four options will be paired differently.</p>"+
    "<p>Please read the labels for each option on each trial carefully to make your choice about "+
    "which option you think is the most rewarding.</p>"+
    "<p></p>"+
    "<p>Whenever you are ready, press the button below to continue. </p>"
}



// ITI screen
var trialSave = function() {
  return "<div>Please wait for the next trial...</div>"
}

// was used to change feedback time.
var getFeedTime = function(){

    var Ftime = 750

  return Ftime
}


// ---------------All Trial Screens, in order seen (sort of)------------------------------------
// Questionnaires and Demographics
var welcome = {
  type: "html-keyboard-response",
  stimulus: "Welcome to our experiment. Press any key to begin.",
  data: {
    Save_Data: 'NoSave'
  }
};

// var getMTurkID = {
//   type: 'survey-text',
//   data: {
//     Save_Data: 'Save'
//   },
//   questions: [{
//     prompt: "Please enter your MTurk ID in the box below. This will be used to give bonuses and for possible notification of future studies.",
//     rows: 1,
//     columns: 30
//   }],
//   on_finish: function(data){
//     console.log(data.responses)
//     turkID = data.responses
//   }
// }

var ageScreen = {
  type: 'survey-text',
  data: {
    Save_Data: 'Save'
  },
  questions: [{
    prompt: "Please type your age in the box",
    rows: 1,
    columns: 5
  }],
  on_finish: function(data){
    //psiturk.recordTrialData(data.responses)
  }
}

var GenderQ = ["Male", "Female", "Prefer not to respond"];
var EthnicityQ = ["Not Hispanic or Latino", "Hispanic or Latino", "Prefer not to answer"];
var RaceQ = ["American Indian or Alaskan Native", "Asian", "Native Hawaiin or Other Pacific Islander", "Black or African American", "White", "More than one Race", "Prefer not to answer"];
var demographicScreen = {
  type: 'survey-multi-choice',
  data: {
    Save_Data: 'Save'
  },
  questions: [{
      prompt: "Please select your gender",
      options: GenderQ,
      required: true
    },
    {
      prompt: "Please select your ethnicity",
      options: EthnicityQ,
      required: true
    },
    {
      prompt: "Please select your race",
      options: RaceQ,
      required: true
    }
  ],
};


var trainingIntro = {
  type: "html-keyboard-response",
  stimulus: "<p>Welcome to our experiment!</p>" +
    "<p>In this study we are interested in how people use information to make " +
    "decisions. On each trial you will be asked to pick from one of two options. "+
    "Each option has a chance to award some points.</p>"+
    "<p>Your job is to figure out which option in each pair is most rewarding.</p>" +
    "<p>To pick an option, just click on the image box of your choice using your mouse. </p>" +
    "<p> </p>" +
    "<p> </p>" +
    "<p>Press any key to begin...</p>",
  data: {
    Save_Data: 'NoSave'
  },
  post_trial_gap: 1000,
  on_finish: function(data) {
    trainShow = 1
    currPhase = 'Train'
  }
};


var trialScreen = {
  type: 'html-button-response',
  data: {
    Save_Data: 'NoSave'
  },
  stimulus: getStim,
  button_html: getButton,
  choices: getChoices,
  post_trial_gap: 0,
  on_finish: function(data) {
    totalTime=jsPsych.data.get().select('rt').values
    var trialTime = totalTime[totalTime.length - 1]
    var keyPressed = data.button_pressed;
    dataMatrix.push(keyPressed);
    reaction.push(trialTime)
  }
};

// Feedback trial screens
var RewardFeed = {
  type: 'html-keyboard-response',
  stimulus: getRewardFeedback,
  data: getDataVal,
  trial_duration: 500,
  post_trial_gap: 0,
  response_ends_trial: false,
  on_load: boxAnim
}

var RewardFeed2 = {
  type: 'html-keyboard-response',
  stimulus: getNextFeed,
  trial_duration: getFeedTime,
  post_trial_gap: 0,
  response_ends_trial: false
}


// Final survey to assess participant knowledge
var finalSurvey = {
  type: 'survey-text',
  questions: [{
    prompt: "Briefly, please describe what you think you did during this experiment.",
    rows: 5,
    columns: 40
  }],
  data: {
    Save_Data: 'Save'
  },
}

// // Signal end of the experiment
var ExpEnd = {
  type: 'html-button-response',
  stimulus: sumPoint,
  choices:['End']
}


var saveData = {
  type: 'html-keyboard-response',
  stimulus: trialSave,
  data: {
    Save_Data: 'NoSave'
  },
  trial_duration: 500,
  post_trial_gap: 0, // Remove this too?
  response_ends_trial: false
}

// // Signal end of the experiment
var testInstrux = {
  type: 'html-button-response',
  stimulus: getTestInstrux,
  choices:['Continue'],
  on_finish: function(data) {
    trainShow = 0
    testShow = 1
    currPhase = 'Test'
  }
}



// Forces fullscreen.
// timeline.push({
//   type: 'fullscreen',
//   fullscreen_mode: true
// });


timeline.push(welcome);

//timeline.push(ageScreen);
//timeline.push(demographicScreen);
//timeline.push(politicalScreen);

timeline.push(trainingIntro);
//for (let i = 0; i < 150; i++) {
for (let i = 0; i < allTrials1.length; i++) {
    if (i == trainTrials1.length){
      timeline.push(testInstrux)
    }
    timeline.push(trialScreen); // Shows trial, then feedback
    timeline.push(RewardFeed);
    timeline.push(RewardFeed2);
    timeline.push(saveData);
}









timeline.push(ExpEnd)