const identityMap = {
    FULL: '正職',
    PART: '兼職'
};
  
const salaryTypeMap = {
    MONTH: '月薪',
    HOUR: '時薪'
};

function getIdentityDisplay(identity) {
    return identityMap[identity] || identity;
}
  
function getSalaryTypeDisplay(salary_type) {
    return salaryTypeMap[salary_type] || salary_type;
}
  
export {
    identityMap,
    salaryTypeMap,
    getIdentityDisplay,
    getSalaryTypeDisplay
};